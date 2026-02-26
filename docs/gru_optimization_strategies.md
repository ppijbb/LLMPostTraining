# Global GRU 연산량 최적화 전략

## 현재 상황 분석

### 현재 설정
```json
{
  "hidden_size": 512,
  "num_experts": 128,
  "router_dim": 128
}
```

**GRU 연산량:**
- Input size: `512`
- Hidden size: `128 * 128 = 16,384` ⚠️ **매우 큼!**
- GRU 파라미터 수: `3 * (512 * 16384 + 16384 * 16384) ≈ 850M` (매우 큼!)
- 연산 복잡도: `O(batch * seq_len * input_size * hidden_size)`

### 문제점
- Hidden dimension이 과도하게 큼 (16K)
- GRU 연산이 전체 모델의 병목
- 메모리 사용량 과다

## 최적화 전략 (우선순위 순)

### 🥇 **1. router_dim 축소 (가장 효과적, 즉시 적용 가능)**

**현재:** `router_dim = 128`
**권장:** `router_dim = 32` 또는 `64`

**효과:**
- Hidden size: `128 * 128 = 16,384` → `128 * 32 = 4,096` (75% 감소)
- 파라미터 수: `850M` → `~200M` (75% 감소)
- 연산량: **4배 감소**

**구현:**
```python
# config에서 router_dim만 변경
"router_dim": 32  # 또는 64
```

**주의사항:**
- router_dim이 너무 작으면 routing 표현력 저하 가능
- 32-64 범위가 일반적으로 충분함
- 실험적으로 최적값 찾기

---

### 🥈 **2. Low-Rank Factorization (GRU Weight 분해)**

**아이디어:** GRU의 큰 weight matrix를 두 개의 작은 matrix로 분해

```python
# 기존: W: [input_size, hidden_size] = [512, 16384]
# 분해: W = U @ V^T
#       U: [512, rank], V: [16384, rank]
#       rank << min(input_size, hidden_size)

# 예: rank = 256
# 파라미터: 512*16384 = 8.4M → 512*256 + 16384*256 = 4.3M (50% 감소)
```

**구현 예시:**
```python
class LowRankGRU(nn.Module):
    def __init__(self, input_size, hidden_size, rank=256):
        super().__init__()
        self.rank = rank
        # Input projection: [input_size, rank]
        self.U_ih = nn.Linear(input_size, rank, bias=False)
        self.U_hh = nn.Linear(hidden_size, rank, bias=False)
        # Output projection: [rank, hidden_size]
        self.V_ih = nn.Linear(rank, hidden_size, bias=False)
        self.V_hh = nn.Linear(rank, hidden_size, bias=False)
        
    def forward(self, x, h):
        # x: [batch, seq, input_size]
        # h: [batch, hidden_size]
        
        # Low-rank projection
        x_proj = self.U_ih(x)  # [batch, seq, rank]
        h_proj = self.U_hh(h)   # [batch, rank]
        
        # Expand and compute gates
        x_gate = self.V_ih(x_proj)  # [batch, seq, hidden_size]
        h_gate = self.V_hh(h_proj)  # [batch, hidden_size]
        
        # GRU gates (simplified)
        # ... (실제 GRU 로직)
```

**효과:**
- 파라미터: 50-75% 감소
- 연산량: 30-50% 감소
- rank 선택이 중요 (256-512 권장)

---

### 🥉 **3. Lightweight GRU (Linear + Gating)**

**아이디어:** Full GRU 대신 단순한 Linear + Gating 구조

```python
class LightweightRouter(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, router_dim):
        super().__init__()
        # 단순 Linear projection
        self.proj = nn.Linear(input_size, num_experts * router_dim, bias=False)
        # Optional: Lightweight gating
        self.gate = nn.Linear(input_size, num_experts * router_dim, bias=False)
        
    def forward(self, x, h_prev=None):
        # x: [batch, seq, input_size]
        proj_out = self.proj(x)  # [batch, seq, num_experts * router_dim]
        
        if h_prev is not None:
            # Simple gating with previous hidden state
            gate_signal = torch.sigmoid(self.gate(x))
            # Residual connection with gating
            output = gate_signal * proj_out + (1 - gate_signal) * h_prev
        else:
            output = proj_out
            
        return output, output  # (output, hidden_state)
```

**효과:**
- 파라미터: 80-90% 감소
- 연산량: 70-85% 감소
- 단순하지만 성능 저하 가능성

---

### 4. **Sparse GRU (Structured Sparsity)**

**아이디어:** GRU weight에 structured sparsity 적용

```python
# Block-sparse 또는 Group-sparse GRU
# 예: 4개 그룹으로 나누어 각 그룹만 활성화
```

**효과:**
- 파라미터: 50-75% 감소
- 연산량: 50-75% 감소
- 구현 복잡도 높음

---

### 5. **Grouped/Block-wise GRU**

**아이디어:** Hidden state를 여러 그룹으로 나누어 각각 작은 GRU 사용

```python
class GroupedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups=4):
        super().__init__()
        self.num_groups = num_groups
        group_size = hidden_size // num_groups
        self.grus = nn.ModuleList([
            nn.GRU(input_size, group_size, batch_first=True)
            for _ in range(num_groups)
        ])
        
    def forward(self, x, h):
        # 각 그룹별로 독립적으로 처리
        outputs = []
        h_outs = []
        for i, gru in enumerate(self.grus):
            h_i = h[:, i*group_size:(i+1)*group_size] if h is not None else None
            out_i, h_i = gru(x, h_i)
            outputs.append(out_i)
            h_outs.append(h_i)
        return torch.cat(outputs, dim=-1), torch.cat(h_outs, dim=0)
```

**효과:**
- 파라미터: 30-50% 감소 (그룹 수에 따라)
- 연산량: 병렬화 가능
- 구현 복잡도 중간

---

## 즉시 적용 가능한 최적화 (우선순위)

### ✅ **1단계: router_dim 축소 (즉시 적용)**

```json
// seqorth_small_config.json
{
  "router_dim": 32  // 128 → 32 (75% 감소)
}
```

**예상 효과:**
- GRU hidden size: 16,384 → 4,096
- 파라미터: 850M → ~200M
- 연산량: **4배 감소**
- 메모리: **4배 감소**

**검증 방법:**
- router_dim을 128 → 64 → 32로 점진적 축소
- Routing 성능 모니터링
- 최적 trade-off 찾기

---

### ✅ **2단계: Low-Rank Factorization (구현 필요)**

현재 코드에 Low-Rank GRU 구현 추가

**구현 위치:**
- `models/seqorth_model.py`의 `SeqorthRouter` 클래스
- `self.load_balancer`를 Low-Rank GRU로 교체

**권장 rank:**
- router_dim=32일 때: rank=128-256
- router_dim=64일 때: rank=256-512

---

### ✅ **3단계: Lightweight Router (성능 검증 후)**

Full GRU 대신 Linear + Gating 구조로 교체

**적용 조건:**
- router_dim 축소 + Low-Rank로도 부족할 때
- 성능 저하가 허용 가능한 범위일 때

---

## 성능 비교 예상

| 방법 | 파라미터 감소 | 연산량 감소 | 구현 난이도 | 성능 영향 |
|------|-------------|-----------|-----------|----------|
| router_dim 축소 | 75% | 75% | ⭐ 매우 쉬움 | 낮음 |
| Low-Rank | 50-75% | 30-50% | ⭐⭐ 쉬움 | 낮음-중간 |
| Lightweight | 80-90% | 70-85% | ⭐⭐ 쉬움 | 중간-높음 |
| Sparse | 50-75% | 50-75% | ⭐⭐⭐ 어려움 | 중간 |
| Grouped | 30-50% | 30-50% | ⭐⭐ 쉬움 | 낮음 |

---

## 권장 실행 계획

### Phase 1: 즉시 적용 (오늘)
1. ✅ `router_dim: 128 → 64` 변경
2. ✅ 학습 시작, 성능 모니터링
3. ✅ 성능 유지 확인 후 `router_dim: 64 → 32` 시도

### Phase 2: 추가 최적화 (1-2일)
1. Low-Rank GRU 구현
2. router_dim=32 + Low-Rank 적용
3. 성능 비교

### Phase 3: 고급 최적화 (필요시)
1. Lightweight Router 구현
2. 성능-효율성 trade-off 최적화

---

## 구현 예시: Low-Rank GRU

```python
class LowRankGRU(nn.Module):
    """
    Low-rank factorization of GRU for efficient routing.
    W = U @ V^T where U: [input_size, rank], V: [hidden_size, rank]
    """
    def __init__(self, input_size, hidden_size, rank=None, num_layers=1, batch_first=True):
        super().__init__()
        if rank is None:
            rank = min(input_size, hidden_size) // 4  # Default: 1/4 of smaller dimension
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Low-rank projections for input-to-hidden
        self.U_ih = nn.Linear(input_size, 3 * rank, bias=False)  # 3 gates
        self.V_ih = nn.Linear(rank, hidden_size, bias=False)
        
        # Low-rank projections for hidden-to-hidden
        self.U_hh = nn.Linear(hidden_size, 3 * rank, bias=False)
        self.V_hh = nn.Linear(rank, hidden_size, bias=False)
        
    def forward(self, x, h=None):
        # x: [batch, seq, input_size] if batch_first else [seq, batch, input_size]
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        batch_size, seq_len, _ = x.shape
        
        if h is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                          device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_size]
            h_t = h[-1]  # [batch, hidden_size]
            
            # Low-rank input projection
            x_proj = self.U_ih(x_t)  # [batch, 3*rank]
            x_proj = x_proj.view(batch_size, 3, self.rank)  # [batch, 3, rank]
            x_gates = self.V_ih(x_proj)  # [batch, 3, hidden_size]
            
            # Low-rank hidden projection
            h_proj = self.U_hh(h_t)  # [batch, 3*rank]
            h_proj = h_proj.view(batch_size, 3, self.rank)  # [batch, 3, rank]
            h_gates = self.V_hh(h_proj)  # [batch, 3, hidden_size]
            
            # GRU gates
            r_gate = torch.sigmoid(x_gates[:, 0] + h_gates[:, 0])  # reset
            z_gate = torch.sigmoid(x_gates[:, 1] + h_gates[:, 1])  # update
            n_gate = torch.tanh(x_gates[:, 2] + r_gate * h_gates[:, 2])  # new
            
            # Update hidden state
            h_t = (1 - z_gate) * n_gate + z_gate * h_t
            h[-1] = h_t
            outputs.append(h_t)
        
        output = torch.stack(outputs, dim=1)  # [batch, seq, hidden_size]
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, h
```

**사용법:**
```python
# 기존 코드 대체
# self.load_balancer = nn.GRU(...)
self.load_balancer = LowRankGRU(
    input_size=self.hidden_size,
    hidden_size=self.num_experts * self.router_dim,
    rank=256,  # 또는 router_dim * 2
    num_layers=1,
    batch_first=True
)
```

---

## 결론

**가장 효과적이고 즉시 적용 가능한 방법:**
1. ✅ **router_dim 축소** (128 → 32): 75% 연산량 감소, 구현 5분
2. ✅ **Low-Rank Factorization**: 추가 50% 감소, 구현 1-2시간

**두 방법을 조합하면:**
- 총 연산량: **87.5% 감소** (4배 × 2배 = 8배)
- 파라미터: **87.5% 감소**
- 메모리: **87.5% 감소**

이 정도면 Global GRU의 연산 비중이 크게 줄어들 것입니다!
