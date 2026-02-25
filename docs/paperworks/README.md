# SPECTRA: Sinkhorn Projected Experts for Consistent TRAjectory Routing for Mixture-of-Experts

논문 초안 작업 디렉토리

## 📁 파일 구성

### 본문 (Main Paper)
1. **01_abstract.txt** - 초록 및 주요 기여
2. **02_introduction.txt** - 서론 및 연구 동기
3. **03_related_work.txt** - 관련 연구
4. **04_method.txt** - SPECTRA 방법론 상세 설명
5. **05_experiments.txt** - 실험 설정 및 평가 방법
6. **06_results.txt** - 실험 결과 및 분석
7. **07_discussion.txt** - 논의 및 한계점
8. **08_conclusion.txt** - 결론 및 향후 연구 방향

### 부록 (Appendix)
9. **09_appendix.txt** - 수학적 세부사항, 추가 실험, 구현 세부사항

---

## 📊 논문 개요

### 핵심 아이디어
OSR (Orthogonal Sinkhorn Routing)과 Gram matrix 기반 직교성 제약을 통해 MoE 모델의 expert 전문화, 다양성, 그리고 최적 부하 분산을 동시에 달성하는 새로운 라우팅 메커니즘. OSR은 학습 파라미터 없이 수학적으로 expert 분리를 보장하는 repulsive cost function을 사용합니다.

### 주요 기여
1. **OSR (Orthogonal Sinkhorn Routing)**: Repulsive cost function을 통한 수학적 expert 분리 보장 (학습 파라미터 0개)
2. **Gram Matrix 기반 Orthogonality Constraints**: Expert 간 직교성을 명시적으로 강제
3. **GRU 기반 Sequential Routing**: 컨텍스트 인식 expert 선택 및 일관된 궤적 생성
4. **Expression Projector**: Expert 전문화 발견을 위한 직교 투영
5. **Comprehensive Ablation Study**: 각 컴포넌트의 기여도 정량화
6. **Modular Implementation**: 모든 HuggingFace 모델에 적용 가능

### 주요 결과 (예상)
- Switch Transformer 대비 **X.X%** 성능 향상
- Expert overlap **XX%** 감소
- Expert collapse 방지 (collapse rate: **0%**)
- 계산 오버헤드 최소화 (**X%**)
- 전문 도메인(코드, 수학, 과학)에서 더 큰 성능 향상

---

## 🔬 연구 방법론

### SPECTRA 구성 요소

```
┌─────────────────────────────────────┐
│          Input Token                │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       │                │
    ┌──▼──┐      ┌──────▼───────┐
    │ GRU │      │ Expression   │
    │     │      │ Projector    │
    └──┬──┘      └──────┬───────┘
       │                │
       └────────┬───────┘
                │
         ┌──────▼─────────┐
         │ Gram Matrix    │
         │ Speciality     │
         │ Penalty        │
         └──────┬─────────┘
                │
         ┌──────▼─────────┐
         │ Cosine         │
         │ Similarity     │
         │ Scoring        │
         └──────┬─────────┘
                │
         ┌──────▼─────────┐
         │ OSR Cost       │
         │ (Repulsion)    │
         └──────┬─────────┘
                │
         ┌──────▼─────────┐
         │ Sinkhorn       │
         │ Optimization   │
         └──────┬─────────┘
                │
         ┌──────▼─────────┐
         │ Top-k          │
         │ Selection      │
         └──────┬─────────┘
                │
         ┌──────▼─────────┐
         │ Expert         │
         │ Execution      │
         └────────────────┘
```

### 수학적 정식화

**Gram Matrix**:
```
G = R · R^T  ∈ ℝ^(E×E)
```

**OSR Repulsive Cost** (replaces separate speciality penalty):
```
Cost = -Similarity + λ · Repulsion
Repulsion = |Similarity| @ (Gram(E_expr) ⊙ (1 - I))²
```

**Sinkhorn Optimization**:
```
Q = Sinkhorn(Cost)  # Doubly stochastic matrix
```

**Routing Weights**:
```
w = topk(Q, k)  # From Sinkhorn output
```

---

## 📈 실험 설정

### 모델
- GPT-2-Medium (345M)
- LLaMA-2-7B
- Mixtral-8x7B (router 교체)

### 데이터셋
**Training**: The Pile (100B tokens)

**Evaluation**:
- Language Understanding: MMLU, HellaSwag, ARC, PIQA, BoolQ
- Code: HumanEval, MBPP
- Math: GSM8K, MATH
- Science: PubMedQA, SciFact

### Baseline
- Switch Transformer (Top-1, Top-2)
- Expert Choice Routing
- Hash Routing
- Dense MLP (upper bound)

### Ablation Variants
- SPECTRA w/o Expression
- SPECTRA w/o GRU (Sequential Router)
- SPECTRA w/o OSR (Repulsive Cost)
- SPECTRA w/o Repulsion (λ=0)
- SPECTRA w/o Orthogonal Constraint

---

## 💻 구현

### 코드 구조
```
models/
├── g3moe_model.py          # 핵심 G3MoE 구현
├── spectra.py         # SPECTRA 라우팅 (모듈화)
├── spectra_ablation.py    # Ablation 변형
└── g3moe_config.py         # 설정 클래스

eval/
├── information_theoretic_analysis.py  # Expert 분석
└── benchmark_runner.py                # 평가 하네스

sft/
├── trainer.py              # 학습 루프
└── config/                 # 학습 설정
```

### 주요 하이퍼파라미터
```yaml
model:
  num_experts: 8
  num_experts_per_tok: 2
  router_dim: 128
  
loss:
  router_entropy_coef: 0.1  # Entropy minimization for sharp routing
  ortho_loss_coef: 0.01      # Optional orthogonal loss on projector weights
  osr_repulsion_weight: 0.5  # Repulsive cost function coefficient

# Note: Unlike traditional MoE, SPECTRA does not require aux_loss_coef or
# speciality_loss_coef. OSR structurally enforces load balancing and expert
# separation without explicit loss terms.
  
optimizer:
  lr_router: 5e-5
  lr_expert: 1e-5
  lr_other: 1e-5
```

---

## 📝 논문 작성 진행 상황

### ✅ 완료
- [x] Abstract
- [x] Introduction
- [x] Related Work
- [x] Method
- [x] Experiments
- [x] Results (템플릿)
- [x] Discussion
- [x] Conclusion
- [x] Appendix

### 🔄 진행 중
- [ ] 실제 실험 실행
- [ ] 결과 데이터 수집
- [ ] 통계 분석
- [ ] 시각화 생성

### 📋 예정
- [ ] LaTeX 변환
- [ ] 그림 및 표 생성
- [ ] 참고문헌 정리
- [ ] 초록 최적화
- [ ] 동료 리뷰
- [ ] 투고 준비

---

## 🎯 투고 목표

### 추천 학회/저널
1. **NeurIPS 2025** (Deadline: ~May 2025)
   - Top-tier ML conference
   - MoE/Efficient models 관련 강세

2. **ICML 2025** (Deadline: ~February 2025)
   - Theory + empirical work 균형
   - Routing mechanism 혁신 강조

3. **ICLR 2026** (Deadline: ~October 2025)
   - Representation learning 관점
   - Orthogonality 이론 강조

4. **JMLR** (Journal, Rolling submission)
   - 긴 형식 논문 가능
   - 포괄적 ablation study 적합

### 예상 논문 길이
- Main paper: ~8-10 pages (conference format)
- With appendix: ~20-25 pages
- Full version (journal): ~35-40 pages

---

## 🔍 핵심 메시지

### 1문장 요약
> Gram matrix 기반 직교성 제약을 통해 MoE expert의 전문화와 다양성을 동시에 달성하는 새로운 라우팅 메커니즘

### 3문장 요약
> 기존 MoE 라우팅은 expert collapse와 전문화 부족 문제로 어려움을 겪는다.
> SPECTRA는 OSR (Orthogonal Sinkhorn Routing)과 Gram matrix를 활용한 직교성 제약, 그리고 GRU 기반 sequential routing을 결합하여 이 문제를 해결한다. OSR은 학습 파라미터 없이 repulsive cost function을 통해 수학적으로 expert 분리를 보장한다.
> 종합적인 ablation study를 통해 각 컴포넌트가 성능에 기여함을 검증하고, 특히 전문 도메인에서 큰 성능 향상을 달성했다.

### 엘리베이터 피치 (30초)
> "MoE 모델의 expert들이 비슷한 기능을 학습하거나 일부만 사용되는 문제가 있습니다.
> 저희는 OSR (Orthogonal Sinkhorn Routing)과 Gram matrix를 사용해 expert들이 직교하도록 강제하고 최적 부하 분산을 달성하는 SPECTRA를 제안합니다. OSR은 학습 파라미터 없이 repulsive cost function을 통해 수학적으로 expert 분리를 보장합니다.
> 이를 통해 각 expert가 코드, 수학, 과학 등 명확한 도메인을 전문화하도록 유도하고,
> Switch Transformer 대비 X%의 성능 향상을 달성했습니다.
> 더불어 모든 HuggingFace 모델에 적용 가능한 모듈화된 구현을 제공합니다."

---

## 📚 참고 자료

### 핵심 선행 연구
1. **Switch Transformer** (Fedus et al., 2021)
   - 표준 MoE routing baseline
   - Top-1 routing + load balancing loss

2. **Mixtral 8x7B** (Jiang et al., 2024)
   - 오픈소스 sparse MoE
   - 실용적 성능 입증

3. **Expert Choice Routing** (Zhou et al., 2022)
   - 역방향 routing 패러다임
   - Load balancing 개선

4. **Sparse Upcycling** (Komatsuzaki et al., 2022)
   - Dense → MoE 변환
   - 효율적 학습 방법

### 수학적 배경
- **Gram-Schmidt Orthogonalization**: 직교 기저 생성
- **Gram Matrix**: 벡터 간 내적 행렬
- **Frobenius Norm**: 행렬 norm 계산

---

## 🔧 실험 체크리스트

### Pre-실험
- [ ] 데이터셋 준비 (The Pile)
- [ ] 평가 벤치마크 설정
- [ ] Baseline 모델 학습
- [ ] Hyperparameter grid search
- [ ] 코드 디버깅 및 검증

### 본 실험
- [ ] GPT-2-Medium 학습 (Switch, SPECTRA, Ablations)
- [ ] LLaMA-2-7B 학습
- [ ] Mixtral-8x7B router 교체
- [ ] 전체 벤치마크 평가
- [ ] Expert 분석 (specialization, usage, etc.)

### Post-실험
- [ ] 통계적 유의성 검정
- [ ] 결과 시각화 (t-SNE, heatmaps, etc.)
- [ ] Error analysis
- [ ] Qualitative examples
- [ ] 최종 성능 검증

### 재현성
- [ ] Random seed 설정
- [ ] 환경 설정 문서화
- [ ] 모든 config 파일 저장
- [ ] Checkpoint 저장
- [ ] Logging 완비

---

## 📊 예상 결과 (템플릿)

실험 완료 후 다음 형식으로 결과 기입:

### Main Results
| Model | MMLU | HellaSwag | HumanEval | GSM8K | Avg |
|-------|------|-----------|-----------|-------|-----|
| Switch Top-2 | XX.X | XX.X | XX.X | XX.X | XX.X |
| **SPECTRA** | **XX.X** | **XX.X** | **XX.X** | **XX.X** | **XX.X** |
| Improvement | +X.X% | +X.X% | +X.X% | +X.X% | +X.X% |

### Expert Specialization
| Metric | Switch | SPECTRA | Improvement |
|--------|--------|----------|-------------|
| Expert Entropy | X.XX | X.XX | +X.X% |
| Expert Overlap | XX% | XX% | -XX% |
| Gram Orthogonality | XX.X | XX.X | +XX% |
| Collapse Rate | XX% | 0% | -XX% |

---

## ✉️ 연락처

실험 진행 및 논문 작성 관련 문의:
- 실험 담당: [이름]
- 논문 작성: [이름]
- 코드 리뷰: [이름]

---

## 📄 라이선스

본 연구 코드 및 논문 초안은 Apache 2.0 라이선스 하에 배포됩니다.

---

## 🙏 감사의 말

- HuggingFace Transformers 팀
- EleutherAI (The Pile 데이터셋)
- 계산 자원 제공: [기관명]
- 논문 리뷰: [리뷰어들]

---

**마지막 업데이트**: 2025-11-11  
**버전**: 0.1 (초안)  
**상태**: 실험 대기 중

