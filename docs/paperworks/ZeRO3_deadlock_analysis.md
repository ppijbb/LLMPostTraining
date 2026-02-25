# DeepSpeed ZeRO-3 Backward Deadlock: 원인 분석

## 1. 가설: "이미지 있는 shard vs 없는 shard → optimizer/collective 불일치"

### 1.1 검색/문서 근거

- **DeepSpeed ZeRO-3 특성**  
  - 파라미터·그래디언트가 rank별로 분할되고, backward 시 **reduce-scatter**로 동기화.  
  - **모든 rank가 같은 collective를 같은 순서로** 호출해야 함. 그렇지 않으면 한 rank가 대기하는 collective에 다른 rank가 참여하지 않아 **deadlock**.

- **PR #5008 (Delay reduce-scatter for ZeRO3 leaf modules)**  
  - 요지: *"Our **data parallel processes may activate different sets of experts**, but the hook is **not fired unless the expert is activated at a forward pass**. **The reduce-scatter is called only on some processes** in this case."*  
  - 즉, **forward에서 서로 다른 모듈이 활성화되면, 일부 프로세스만 reduce-scatter에 참여** → collective 불일치 → deadlock과 같은 현상.

- **ZeRO-3와 다른 computation path**  
  - backward에서 rank마다 **다른 computation path**를 타면,  
    - 서로 다른 파라미터에 gradient가 생기거나  
    - reduce-scatter / AllGather 참여 순서·집합이 달라질 수 있음  
  → collective 순서/참여 불일치 → deadlock 위험.

이를 VLM의 “이미지 유무”에 그대로 적용하면:

- **이미지 있는 shard (rank A)**  
  - forward: `pixel_values` → **vision encoder** → image features → **merge(이미지+텍스트)** → LLM  
- **이미지 없는 shard (rank B)**  
  - forward: 텍스트만 → **LLM** (vision, merge 등 **미경유**)

→ **같은 step에서 rank별로 지나가는 모듈/연산이 다름** → backward graph와 “어떤 파라미터가 gradient를 받는지”가 rank마다 달라질 수 있음 → **가설(optimizer/collective 쪽에서 뻑 난다)은 타당**.

---

## 2. "Vision tower는 학습에서 제외(freeze)되어 있는데?"

### 2.1 Freeze만으로는 부족한 이유

- **Forward graph 자체가 다름**  
  - freeze는 “vision **파라미터**에 gradient를 안 쌓는다”일 뿐,  
    **forward에서 vision을 타느냐 안 타느냐**는 별개.
  - 이미지 있는 rank: **vision을 반드시 통과** (이미지 → feature 생성).  
  - 이미지 없는 rank: **vision을 아예 사용하지 않음**.  
  → **연산 그래프(어떤 module이 forward에 참여하는지)가 rank마다 다름.**

- **Backward에서의 차이**  
  - Vision이 freeze라도, 이미지 rank에서는  
    - `image_features`까지는 gradient가 전파되고  
    - vision 자체는 `requires_grad=False`라 vision **파라미터**에는 grad가 없음.  
  - 대신 **vision을 거쳐 나온 tensor를 쓰는 이후 모듈(merge, LLM 입력 등)** 은  
    - 이미지 rank: “image_features + text” 구조로 backward  
    - 텍스트 rank: “text만” 구조로 backward  
  → **같은 LLM이라도, backward가 도달하는 연산/버퍼/순서가 다를 수 있음.**

- **Qwen3‑VL 구조에서의 차이**  
  - `get_placeholder_mask` / `masked_scatter` 등 **이미지가 있을 때만** 호출되는 분기 존재.  
  - 이미지 rank: 이 경로가 **반드시** forward/backward에 포함.  
  - 텍스트 rank: 이 경로 **자체가 없음**.  
  → **모듈·연산 집합이 rank마다 다르다**는 점은 vision freeze와 무관하게 유지.

- **ZeRO-3 reduce-scatter / hook**  
  - Hook은 “해당 **파라미터가 backward에 참여했을 때”** reduce-scatter를 부름.  
  - Vision은 freeze라 **vision 파라미터**에 대한 reduce-scatter는 양쪽 다 없을 수 있음.  
  - 하지만  
    - **이미지유무에 따라 backward가 지나가는 모듈·연산이 다르고**  
    - **다른 파라미터(예: merge 근처, 또는 이미지 분기와 연결된 레이어)가 rank별로 다르게 참여**하거나,  
    - **AllGather / release 순서**가 다르게 잡히면,  
  → **collective 순서/참여가 어긋날 수 있어** optimizer 단에서 “뻑”이 나거나 deadlock처럼 보일 수 있음.

정리하면, **“vision이 학습에서 제외”라도, “이미지 유무에 따른 **전체 computation path 차이**”는 그대로라,  
**가설(어느 shard는 이미지 있고 어느 shard는 없어서 연산/optimizer/collective에서 문제)이 맞을 가능성이 높다.**

---

## 3. 대응 방향 (이미 트랜스크립트/구현에서 쓰인 것과 일치)

- **“모든 샘플이 동일한 computation path”**  
  - 트랜스크립트:  
    - *"ZeRO-3 Deadlock 방지 → **모든 배치가 동일한 computation path**"*  
    - *"Required for VLM training and **ZeRO-3 deadlock prevention**"*  
    - *"텍스트 전용 배치 - **Dummy 이미지 주입** (ZeRO-3 deadlock 방지)"*  
  - 즉, **텍스트 전용이어도**  
    - (가능하면) **동일한 VLM 경로**를 타게 하고  
    - **dummy image / dummy image token**을 넣어,  
    - **모든 rank가 “이미지가 있는 것처럼” 같은 forward/backward 구조**를 갖게 하는 전략.

- **현재 `dataset_utils` / `ensure_vlm_format`**  
  - 이미지 placeholder가 **없는** 텍스트 샘플에는 **이미지를 넣지 않고**,  
    processor에서 `images=None`으로 처리.  
  - 그 결과,  
    - 일부 배치는 `pixel_values` 있음 (이미지 O),  
    - 일부는 `pixel_values` 없음 (이미지 X)  
  → **step/rank별로 “이미지 있음/없음”이 갈리기 쉬움** → **computation path 불일치 → deadlock 가설과 맞음.**

- **수정 제안 (트랜스크립트·문서와 정렬)**  
  - **옵션 A: “항상 동일한 path””**  
    - **모든 배치**에서  
      - `pixel_values`를 반드시 넣고 (실제 이미지 또는 **dummy**),  
      - 텍스트 전용도 **같은 VLM 입력 형태**(이미지 placeholder + dummy image)로 통일.  
    - 그러면 **어느 rank라도**  
      - vision (또는 이미지 branch) → merge → LLM  
      - 같은 순서로 forward/backward  
    → ZeRO-3 collective 순서·참여가 맞을 가능성이 높음.  
  - **옵션 B: “텍스트만” path로 완전 분리**  
    - 이미지 있는 샘플과 **완전히 분리된 데이터셋/배치**를 만들고,  
      - 이미지 배치 step: 모든 rank가 **항상 이미지 포함**  
      - 텍스트만 배치 step: **모든 rank가 텍스트만**  
    - 즉, **한 step 안에서는 모든 rank가 같은 path**가 되게 함.  
  - **옵션 C: DataLoader 수준 통일**  
    - `DistributedSampler`만이 아니라,  
      - **배치를 구성할 때** “이 step은 전부 이미지 포함” vs “전부 텍스트만”이 **전체 rank에서 일치**하도록  
      - (예: 동일한 “배치 타입” 플래그를 rank 0이 정하고 broadcast)  
    - 해서, **최소한 같은 step에서 rank별 path가 갈리지 않게** 함.

---

## 4. 결론

| 항목 | 내용 |
|------|------|
| **가설** | 일부 shard만 이미지 있고, 일부는 없어서 **연산 path·collective 참여가 다르다 → ZeRO-3 backward에서 deadlock/optimizer 쪽 오류** |
| **근거** | DeepSpeed PR #5008, ZeRO-3 collective/동기화 요구, VLM의 `pixel_values` 유무 분기 |
| **Vision freeze** | Vision을 안 넣어도, **이미지 유무에 따른 forward/backward path 차이**는 그대로라, **가설을 무효화하지 않음**. |
| **대응** | **모든 rank가 같은 step에서 동일한 computation path**를 갖도록,  
  - dummy 이미지로 “항상 이미지 있는 path” 통일, 또는  
  - “이미지만 / 텍스트만” 단위로 step을 완전히 나누어 path 통일 |

따라서 **“vision tower는 학습에서 제외되어 있는데 그게 진짜 원인일까?”**에 대한 답은:  
**Vision이 학습에서 제외되어 있어도, 이미지 유무에 따른 path 차이로 인한 ZeRO-3 deadlock/collective 불일치는 발생할 수 있고, 사용자가 겪는 현상의 원인일 가능성이 높다.**
