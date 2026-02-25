# MoE Routing Metrics - 통합 참고 자료 (2025년 11월 28일 기준)

> **⚠️ 중요**: 이 논문의 핵심은 **routing 방법론**입니다. 벤치마크 성능은 보조 지표이며, **routing 자체의 특성과 효과**를 증명하는 것이 우선입니다.

---

## 📅 업데이트 정보
**최종 업데이트**: 2025년 11월 28일
**기준**: 실제 논문과 Technical Report만 인용

---

## 🎯 Routing 방법론 논문의 핵심 지표 (최우선)

### A. Routing Quality Metrics (필수)

#### A.1 Expert Specialization (핵심 주장)
- **Expert Overlap**: Jaccard similarity between expert token sets
  - 낮을수록 specialization 우수
  - **목표**: SPECTRA < Switch Top-2 < Switch Top-1
- **Gram Matrix Orthogonality**: `1 - ||G-I||_F / (E*√2)` (높을수록 좋음, 1.0 = 완전 orthogonal)
  - **현재 측정값**: 0.94 ✅
- **Expert Diversity Score**: 1 - mean(expert_similarity)
  - 높을수록 diverse/specialized
- **Expert-Task Correlation**: Expert별 task specialization score

#### A.2 Load Balancing (필수 비교 지표)
- **Expert Entropy**: H(expert) = -Σᵢ pᵢ log pᵢ
  - 높을수록 균형 (이상: log(E))
- **Load Balancing Coefficient (CV)**: std / mean
  - 낮을수록 균형
  - **현재 측정값**: 0.3 ❌ (목표: < 0.1)
- **Expert Collapse Rate**: 사용되지 않는 expert 비율
- **MaxVio (Maximum Violation)**: max deviation from mean load
- **Gini Coefficient**: Load distribution inequality (0 = perfect equality)
  - **LPR 보고값**: 0.035 (from 0.70)

#### A.3 Routing Decision Quality
- **Routing Entropy**: Per-token routing entropy
- **Routing Consistency**: Checkpoint 간 routing 일관성 (%)
- **Sequential Routing Consistency**: 연속 토큰의 expert 선택 일관성
- **Top-k Overlap**: 연속 토큰의 top-k expert 겹침 비율

#### A.4 Expression Projection Effectiveness
- **Expression-Routing Alignment**: Expression과 routing의 일치도
- **Expression Projection Orthogonality**: Expression projector의 orthogonal quality
- **Ablation Impact**: Expression 제거 시 성능 저하

---

## 📊 실제 논문/Technical Report 기반 모델 및 방법론

### 최신 MoE 모델 (2025)

#### 1. Kimi K2
**논문**: arxiv:2507.20534, "Kimi K2: Open Agentic Intelligence"
**URL**: https://arxiv.org/abs/2507.20534

**Architecture**:
- 1 trillion total parameters
- 32 billion activated per token
- 384 experts total, 8 activated per token (+ shared expert)
- MuonClip optimizer with QK-clip technique
- No expert grouping (n_group = 1)

**Routing Mechanism**:
- QK-clip technique for stable attention and balanced routing
- **⚠️ Routing metrics (CV, Orthogonality, Overlap)는 논문에서 보고되지 않음**

**Performance**:
- SOTA among open-source non-thinking models
- Strong performance in agentic tasks

---

#### 2. GLM-4.5
**Technical Report**: arxiv:2508.06471
**URL**: https://arxiv.org/abs/2508.06471

**Architecture**:
- 355 billion total parameters
- 32 billion activated per token
- Multi-stage training (23 trillion tokens)

**Routing Mechanism**:
- **Loss-free balance approach with sigmoid gating**
- Even distribution across experts
- **⚠️ Routing metrics는 technical report에서 명시적으로 보고되지 않음**

**Performance**:
- TAU-Bench: 70.1%
- AIME 24: 91.0%
- SWE-bench Verified: 64.2%

---

#### 3. Minimax-Text-01 / MiniMax-M1
**Technical Report**: 
- MiniMax-Text-01: arxiv:2501.08313
- MiniMax-M1: arxiv:2506.13585

**Architecture**:
- 456 billion total parameters
- 45.9 billion activated per token
- 32 experts
- ABAB pattern: Alternating Lightning Attention and Softmax Attention
- Context window: 1M tokens (training), 4M tokens (inference)

**Routing Mechanism**:
- MoE routing integrated with ABAB attention pattern
- **⚠️ MoE routing metrics에 대한 구체적 수치는 technical report에서 보고되지 않음**

---

#### 4. DeepSeek-V3
**Technical Report**: DeepSeek official

**Architecture**:
- 671 billion total parameters
- 37 billion activated per token
- 256 routed experts + 1 shared expert per layer
- 8 routed + 1 shared expert activated per token

**Routing Mechanism**:
- **Auxiliary-loss-free load balancing**
- Dynamic bias adjustment (underutilized → bias increases, overutilized → bias decreases)
- Sequence-wise balance loss (α = 0.0001)

**⚠️ 주의**: 구체적인 routing metrics (CV, Orthogonality)는 technical report에서 명시적으로 보고되지 않음

---

#### 5. Qwen3-MoE
**Technical Report**: arxiv:2505.09388, "Qwen3 Technical Report"
**URL**: https://arxiv.org/abs/2505.09388

**Architecture**:
- 128 total experts
- 8 experts activated per token
- No shared experts (unlike Qwen2.5-MoE)
- Fine-grained expert segmentation

**Routing Mechanism**:
- **Global-batch load balancing loss**
- Top-k learned gating function (k=8)

**⚠️ 주의**: 구체적인 routing metrics (CV, Orthogonality, Overlap)는 technical report에서 명시적으로 보고되지 않음

---

### 최신 Routing 방법론 (2025년 11월 기준)

#### 1. ERMoE (Eigen-Reparameterized MoE)
**논문**: arxiv:2511.10971, November 2025
**제목**: "ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization"
**URL**: https://arxiv.org/abs/2511.10971

**핵심 아이디어**:
- Learned orthonormal eigenbasis for each expert
- Eigenbasis Score = cosine similarity between input features and expert's basis
- Content-aware routing tied directly to experts' representation spaces

**장점**:
- Explicit balancing losses 불필요
- Stable utilization, interpretable specialization
- Natural flatter expert load distributions
- No interference gradients from auxiliary losses

**성능** (논문에서 보고):
- SOTA accuracy on ImageNet classification
- SOTA on cross-modal image-text retrieval (COCO, Flickr30K)
- 3D MRI variant: +7% brain age prediction accuracy

**⚠️ Metrics**: 논문에서 "natural flatter expert load distributions" 언급되나, 구체적인 CV, Orthogonality, Overlap 수치는 보고되지 않음

---

#### 2. Latent Prototype Routing (LPR)
**논문**: arxiv:2506.21328, June 2025
**제목**: "Latent Prototype Routing: Achieving Near-Perfect Load Balancing in Mixture-of-Experts"
**URL**: https://arxiv.org/abs/2506.21328

**핵심 아이디어**:
- Clustering perspective for expert routing
- Generalizes existing routing methods

**실제 보고된 Metrics** (논문에서):
- **Gini coefficient**: 0.70 → 0.035 (average reduction)
- **Min-max expert load ratio**: 1e-6 → 0.70
- **테스트 모델**: DeepSeek-V3, Qwen3-MoE, Mixtral

**⚠️ 주의**: CV, Orthogonality, Expert Overlap은 논문에서 명시적으로 보고되지 않음

---

#### 3. LASER (Load-Aware Scalable Expert Routing)
**논문**: arxiv:2510.03293, October 2025
**제목**: "From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing"
**URL**: https://arxiv.org/abs/2510.03293

**핵심 아이디어**:
- Plug-and-play inference-time routing algorithm
- Adapts to gate's score distribution
- Routes to least-loaded experts when scores are uniform

**특징**:
- No model retraining required
- Inference-time optimization only

**성능** (논문에서 보고):
- Enhanced throughput on Mixtral-8x7B
- Maintains accuracy while improving load balance

**⚠️ Metrics**: 구체적인 CV, Orthogonality 수치는 논문에서 보고되지 않음

---

#### 4. RoMA (Routing Manifold Alignment)
**논문**: arxiv:2511.07419, November 2025
**제목**: "Routing Manifold Alignment Improves Generalization of Mixture-of-Experts LLMs"
**URL**: https://arxiv.org/abs/2511.07419

**핵심 아이디어**:
- Aligns routing weights with task embeddings
- Manifold regularization term
- Lightweight fine-tuning of routers only

**성능** (논문에서 보고):
- Substantial improvements across benchmarks
- Better generalization performance

**⚠️ Metrics**: Routing metrics (CV, Orthogonality)는 논문에서 보고되지 않음

---

#### 5. Advancing Expert Specialization
**논문**: arxiv:2505.22323, May 2025
**제목**: "Advancing Expert Specialization for Better MoE"
**URL**: https://arxiv.org/abs/2505.22323

**핵심 아이디어**:
- Orthogonality loss: Encourages experts to process distinct token types
- Variance loss: Promotes discriminative routing decisions

**성능** (논문에서 보고):
- Performance gains up to 23.79% over classic MoE baselines
- Maintains load balancing without architectural modifications

**⚠️ Metrics**: 구체적인 CV, Orthogonality 수치는 논문에서 보고되지 않음

---

#### 6. Loss-Free Balancing
**논문**: arxiv:2408.15664, August 2024
**제목**: "Loss-Free Load Balancing for Mixture-of-Experts"
**URL**: https://arxiv.org/abs/2408.15664

**핵심 아이디어**:
- Dynamic expert bias adjustment
- No auxiliary losses (eliminates interference gradients)
- Expert-wise bias updated based on recent load

**성능** (논문에서 보고):
- Better performance and load balance
- Tested on MoE models up to 3B parameters
- Trained on up to 200B tokens

**⚠️ Metrics**: 구체적인 CV, Orthogonality 수치는 논문에서 보고되지 않음

---

### 레거시 방법론 (참고용)

#### 1. Switch Transformer
**논문**: Fedus et al., 2021, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"

**실제 보고된 Metrics**:
- **Balanced expert utilization ratio**: 94.8% (nearly uniform distribution)
- **Auxiliary loss coefficient (λ)**: 0.01
- **Training speedup**: Up to 7× compared to dense models

**⚠️ 주의**: CV, Orthogonality, Expert Overlap은 논문에서 명시적으로 보고되지 않음

**Routing Mechanism**:
- Top-1 gating (each token → single expert)
- Auxiliary load balancing loss
- Expert capacity factor (default 1.0)

---

#### 2. Expert Choice Routing
**논문**: Zhou et al., 2022 (NeurIPS), "Mixture-of-Experts with Expert Choice Routing"

**실제 보고된 Metrics**:
- **Training convergence**: More than 2× faster than Switch Transformer/GShard
- **GLUE/SuperGLUE**: 7 out of 11 tasks outperform T5 dense model
- **Load balancing**: Fixed bucket size per expert (balanced by design)

**⚠️ 주의**: 구체적인 CV, Orthogonality, Overlap 수치는 논문에서 보고되지 않음

**Routing Mechanism**:
- Experts select top-k tokens (inverse of token choice)
- Fixed number of tokens per expert
- Variable number of experts per token

---

#### 3. Hash Routing
**논문**: Roller et al., 2021

**실제 보고된 Metrics**:
- **Loss improvement**: 1.5% over dense (16 experts)
- **Load balance**: Perfect (deterministic)
- **Limitation**: No specialization (context ignored)

**⚠️ 주의**: Orthogonality는 N/A (deterministic routing)

---

## 📈 실제 논문에서 보고된 Metrics 요약

### Load Balancing Metrics

| Method | Metric | Value | Source | Note |
|--------|--------|-------|--------|------|
| Switch Transformer | Balanced utilization | 94.8% | Fedus et al., 2021 | Actual reported |
| LPR | Gini coefficient | 0.70 → 0.035 | arxiv:2506.21328 | Actual reported |
| LPR | Min-max expert load ratio | 1e-6 → 0.70 | arxiv:2506.21328 | Actual reported |
| Expert Choice | Training convergence | 2× faster | Zhou et al., 2022 | Actual reported |
| Hash Routing | Load balance | Perfect (CV ~0.0) | Roller et al., 2021 | By design |

### Expert Specialization Metrics

| Method | Metric | Value | Source | Note |
|--------|--------|-------|--------|------|
| Advancing Expert Specialization | Performance gain | Up to 23.79% | arxiv:2505.22323 | Actual reported |
| ERMoE | Load distribution | "Natural flatter" | arxiv:2511.10971 | Qualitative only |
| Hash Routing | Specialization | None (high overlap) | Roller et al., 2021 | By design |

### ⚠️ 중요 발견

**대부분의 논문과 technical report에서 전통적인 routing metrics (CV, Orthogonality, Expert Overlap)를 명시적으로 보고하지 않음**

**실제 수치가 보고된 것**:
- Switch Transformer: Balanced utilization 94.8%
- LPR: Gini coefficient 0.035, Min-max ratio 0.70
- Expert Choice: 2× faster convergence
- Advancing Expert Specialization: 23.79% performance gain

**정량적 수치가 없는 것**:
- ERMoE: "Natural flatter load" (정량적 수치 없음)
- DeepSeek-V3: "Balanced utilization" (정량적 수치 없음)
- Qwen3-MoE: "Global-batch load balancing" (정량적 수치 없음)
- Kimi K2, GLM-4.5, Minimax: Routing metrics 미보고

---

## 📝 논문 표 작성 가이드 (실제 보고된 수치만)

### Table 1: Routing Quality Comparison (핵심 표)

**⚠️ 중요**: 대부분의 최신 논문에서 전통적인 metrics (CV, Orthogonality, Overlap)를 명시적으로 보고하지 않음

```
Method | Expert Overlap | Gram Ortho* | Expert Entropy | Load Balance CV | Gini Coeff | Routing Consistency | Collapse | Source
-------|----------------|-------------|----------------|-----------------|------------|---------------------|----------|-------
Switch Top-1 (2021) | N/A | N/A | N/A | N/A | N/A | N/A | Yes | Fedus et al., 2021 (utilization 94.8% only)
Switch Top-2 (2021) | N/A | N/A | N/A | N/A | N/A | N/A | Partial | Fedus et al., 2021
Expert Choice (2022) | N/A | N/A | N/A | N/A | N/A | N/A | Minimal | Zhou et al., 2022 (2× faster convergence)
Hash Routing | High | N/A | 2.8-3.0 | ~0.0 | ~0.0 | N/A | No | Roller et al., 2021 (by design)
DeepSeek-V3 (2024) | N/A | N/A | N/A | N/A | N/A | N/A | No | Technical Report (metrics not reported)
Qwen3-MoE (2024) | N/A | N/A | N/A | N/A | N/A | N/A | No | arxiv:2505.09388 (metrics not reported)
Kimi K2 (2025) | N/A | N/A | N/A | N/A | N/A | N/A | No | arxiv:2507.20534 (metrics not reported)
GLM-4.5 (2025) | N/A | N/A | N/A | N/A | N/A | N/A | No | arxiv:2508.06471 (metrics not reported)
Minimax (2025) | N/A | N/A | N/A | N/A | N/A | N/A | No | arxiv:2501.08313 (metrics not reported)
ERMoE (2025) | N/A | N/A | N/A | N/A | N/A | N/A | No | arxiv:2511.10971 (metrics not reported)
LPR (2025) | N/A | N/A | N/A | N/A | 0.035 | N/A | No | arxiv:2506.21328 (Gini only)
LASER (2025) | N/A | N/A | N/A | N/A | N/A | N/A | No | arxiv:2510.03293 (metrics not reported)
RoMA (2025) | N/A | N/A | N/A | N/A | N/A | N/A | No | arxiv:2511.07419 (metrics not reported)
SPECTRA (Ours) | 측정 필요 | 0.94 ✅ | 측정 필요 | 0.3 ❌ | 측정 필요 | 측정 필요 | No | 직접 측정
```

**실제 논문에서 보고된 Metrics**:
- **Switch Transformer**: Balanced utilization 94.8% (Fedus et al., 2021)
- **LPR**: Gini coefficient 0.035, Min-max ratio 0.70 (arxiv:2506.21328)
- **Expert Choice**: 2× faster convergence (Zhou et al., 2022)
- **Advancing Expert Specialization**: Up to 23.79% performance gain (arxiv:2505.22323)

**⚠️ 결론**: 대부분의 최신 논문이 전통적인 routing metrics를 보고하지 않으므로, **직접 측정하여 비교**해야 함

**지표 설명**: 
- **Expert Overlap**: Jaccard similarity (낮을수록 좋음, 0% = 완전 분리)
- **Gram Ortho***: `1 - ||G-I||_F / (E*√2)` (높을수록 좋음, 1.0 = 완전 orthogonal)
  - 현재 측정값 0.94: ✅ 좋은 수준
- **Expert Entropy**: Normalized entropy (높을수록 좋음, 3.0 = 완전 균형 for 8 experts)
- **Load Balance CV**: Coefficient of variation (낮을수록 좋음, 0 = 완전 균형)
  - 현재 측정값 0.3: ❌ Moderate imbalance (목표: < 0.1, LPR Gini 0.035 기준)
- **Gini Coefficient**: Load distribution inequality (0 = perfect equality, 1 = perfect inequality)
  - LPR 보고값: 0.035 ✅ (from 0.70)
- **Min-max Expert Load Ratio**: min(expert_load) / max(expert_load) (높을수록 좋음, 1.0 = perfect balance)
  - LPR 보고값: 0.70 ✅ (from 1e-6)
- **Utilization**: Balanced expert utilization ratio
  - Switch Transformer 보고값: 94.8% ✅

---

### Table 2: Ablation Study - Routing Metrics

**⚠️ 주의**: 아래 수치는 목표값/기대값 범위입니다. 실제 실험 후 측정값으로 대체 필요

```
Variant | Expert Overlap | Gram Ortho | Load Balance CV | Routing Consistency | Sequential Consistency
--------|----------------|------------|-----------------|---------------------|------------------------
SPECTRA-Full | 측정 필요 | 0.94 ✅ | 0.3 ❌ | 측정 필요 | 측정 필요
  -Expression | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
  -GRU | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
  -SpecialityPenalty | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
  -OrthoConstraint | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
  -All | 측정 필요 | N/A | 측정 필요 | 측정 필요 | 측정 필요
```

**해석**:
- 각 component 제거 시 routing metrics 변화 측정 필요
- Expression, GRU, SpecialityPenalty, OrthoConstraint의 기여도 정량화

---

### Table 3: Task Performance Benchmarks (보조 표)

```
Model | MMLU | HellaSwag | ARC-C | PIQA | BoolQ | Avg
------|------|-----------|-------|------|-------|-----
Dense MLP | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
Switch Top-1 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
Switch Top-2 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
Expert Choice | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
Hash Routing | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
SPECTRA (Ours) | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
```

**참고**: Task performance는 routing quality의 결과로 보고

---

### Table 4: Specialized Domains (보조 표)

```
Model | HumanEval | MBPP | GSM8K | MATH | PubMedQA | SciFact
------|-----------|------|-------|------|----------|--------
Switch Top-2 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
SPECTRA (Ours) | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요 | 측정 필요
Improvement | 계산 필요 | 계산 필요 | 계산 필요 | 계산 필요 | 계산 필요 | 계산 필요
```

---

## 🎯 SPECTRA 비교 기준 (실제 보고된 수치 기준)

### 실제 측정 가능한 비교 대상

1. **LPR (arxiv:2506.21328)**:
   - Gini coefficient: 0.035 (목표)
   - Min-max expert load ratio: 0.70 (목표)
   - ⚠️ CV, Orthogonality, Overlap은 보고되지 않음

2. **Switch Transformer (Fedus et al., 2021)**:
   - Balanced utilization: 94.8% (참고)
   - ⚠️ CV, Orthogonality, Overlap은 보고되지 않음

3. **Expert Choice (Zhou et al., 2022)**:
   - Training convergence: 2× faster (참고)
   - ⚠️ CV, Orthogonality, Overlap은 보고되지 않음

4. **ERMoE (arxiv:2511.10971)**:
   - "Natural flatter load" (정량적 수치 없음)
   - SOTA performance on ImageNet, COCO

5. **Advancing Expert Specialization (arxiv:2505.22323)**:
   - Up to 23.79% performance gain
   - Orthogonality loss + Variance loss 사용

### 전통적인 Metrics (CV, Orthogonality, Overlap)

**⚠️ 문제**: 대부분의 최신 논문에서 이 metrics를 보고하지 않음

**가능한 접근**:
1. **자체 측정**: 동일한 baseline에서 직접 측정하여 비교
2. **논문 재분석**: 논문의 figure나 appendix에서 추출 가능한 정보 확인
3. **공개 코드**: GitHub repository에서 metrics 계산 코드 확인

---

## 📊 평가해야 할 벤치마크 목록 (보조 지표)

### 1. Language Understanding
- **MMLU** (Massive Multitask Language Understanding): 57개 주제, 5-shot
- **HellaSwag**: 상식 추론, 10-shot
- **ARC-Challenge**: 과학 질문, 25-shot
- **PIQA**: 물리적 추론, 0-shot
- **BoolQ**: Yes/No 질문, 0-shot

### 2. Language Generation
- **WikiText-103**: Language modeling perplexity
- **LAMBADA**: 장거리 의존성 평가
- **TruthfulQA**: 진실성 있는 생성 평가

### 3. Code Understanding
- **HumanEval**: Python 코드 생성 (Pass@1, Pass@10, Pass@100)
- **MBPP**: 기본 프로그래밍 문제

### 4. Mathematical Reasoning
- **GSM8K**: 초등 수학 문제, 8-shot
- **MATH**: 경쟁 수준 수학, 4-shot

### 5. Specialized Domains
- **PubMedQA**: 생의학 질문 답변
- **SciFact**: 과학적 주장 검증

---

## 🔍 비교해야 할 레퍼런스 모델 및 공개 성능 지표

### 1. Mixtral 8x7B
**논문**: Jiang et al., 2024 (Mistral AI)

**공개된 성능 지표**:
| 벤치마크 | 점수 |
|---------|------|
| **MMLU** | 71.34% |
| **GSM8K** | 66.82% |
| **HumanEval** | 40.9% (Pass@1) |

**추가 정보**:
- 8개 expert, 2개 활성화 (top-2 routing)
- 총 파라미터: ~47B (active: ~13B)

---

### 2. DeepSeek-V3
**Technical Report**: DeepSeek official

**공개된 성능 지표**:
| 벤치마크 | 점수 |
|---------|------|
| **MMLU** | 83.7% (37B active) |
| **GSM8K** | 91.3% |

**추가 정보**:
- 671B total, 37B active parameters
- 256 routed experts + 1 shared expert

---

### 3. LLaMA-2 7B (Dense Baseline)
**논문**: Touvron et al., 2023 (Meta)

**공개된 성능 지표**:
| 벤치마크 | 점수 |
|---------|------|
| **MMLU** | 44.4% |
| **HellaSwag** | 77.1% |
| **ARC-Challenge** | 43.2% |
| **GSM8K** | 16.0% |
| **HumanEval** | 11.6% (Pass@1) |

**용도**: Dense baseline으로 사용 (upper bound 비교)

---

### 4. GPT-2-Medium (Dense Baseline)
**논문**: Radford et al., 2019

**공개된 성능 지표**:
| 벤치마크 | 점수 |
|---------|------|
| **WikiText-2 Perplexity** | 22.76 |
| **WikiText-103 Perplexity** | 26.37 |
| **LAMBADA Perplexity** | 15.60 |
| **LAMBADA Accuracy** | 55.48% |

**용도**: Small-scale dense baseline

---

## 🔬 실험 설계 체크리스트 (Routing 중심)

### 1. Baseline Routing 구현 (최우선)
- [ ] Switch Top-1 routing 구현
- [ ] Switch Top-2 routing 구현
- [ ] Expert Choice routing 구현
- [ ] Hash routing 구현
- [ ] Dense MLP baseline (upper bound)

### 2. Ablation Variants
- [ ] SPECTRA-Full (baseline)
- [ ] SPECTRA w/o Expression
- [ ] SPECTRA w/o GRU
- [ ] SPECTRA w/o Speciality Penalty
- [ ] SPECTRA w/o Orthogonal Constraint
- [ ] SPECTRA w/o All Enhancements

### 3. Model Scales
- [ ] GPT-2-Medium (345M) - Dense to MoE
- [ ] LLaMA-2-7B - Dense to MoE
- [ ] Mixtral-8x7B - Router replacement

### 4. Routing Metrics Evaluation Setup (최우선)
- [ ] Expert specialization analysis tools (spectra_analysis.py)
- [ ] Load balancing metrics collection
- [ ] Routing consistency measurement
- [ ] Sequential routing pattern analysis
- [ ] Expression projection quality analysis
- [ ] Training dynamics tracking (over time)

### 5. Task Performance Evaluation Setup (보조)
- [ ] lm-evaluation-harness 설정
- [ ] Custom evaluation scripts (HumanEval, MBPP)
- [ ] Perplexity evaluation setup

### 6. Metrics Collection (우선순위 순)
- [ ] **Routing metrics 자동 수집** (최우선)
  - Expert specialization metrics
  - Load balancing metrics
  - Routing decision quality
  - Expression projection effectiveness
- [ ] **Training dynamics logging** (중요)
  - Time-series data for all routing metrics
- [ ] Task performance metrics 자동 수집 (보조)
- [ ] Computational efficiency metrics 자동 수집 (보조)

---

## 📚 참고 논문 목록 (실제 arxiv 번호)

### 최신 MoE 모델 (2025)
1. **Kimi K2**: arxiv:2507.20534
2. **GLM-4.5**: arxiv:2508.06471
3. **Minimax-Text-01**: arxiv:2501.08313
4. **MiniMax-M1**: arxiv:2506.13585
5. **Qwen3-MoE**: arxiv:2505.09388

### 최신 Routing 방법론 (2025)
1. **ERMoE**: arxiv:2511.10971
2. **LPR**: arxiv:2506.21328
3. **LASER**: arxiv:2510.03293
4. **RoMA**: arxiv:2511.07419
5. **Advancing Expert Specialization**: arxiv:2505.22323
6. **Local Routing Consistency**: arxiv:2505.16056
7. **Input Domain Aware MoE**: arxiv:2510.16448
8. **GRACE-MoE**: arxiv:2509.25041
9. **MaxScore**: arxiv:2508.12801
10. **Loss-Free Balancing**: arxiv:2408.15664

### 레거시 방법론
1. **Switch Transformer**: Fedus et al., 2021
2. **Expert Choice**: Zhou et al., 2022 (NeurIPS)
3. **Mixtral**: Jiang et al., 2024
4. **Hash Routing**: Roller et al., 2021

---

## ⚠️ 중요 발견 및 권장사항

### 문제점
**대부분의 최신 MoE 모델과 routing 방법론이 전통적인 routing metrics (CV, Orthogonality, Expert Overlap)를 논문/technical report에서 명시적으로 보고하지 않음**

### 실제 보고된 Metrics (2025년 11월 기준)
1. **LPR (arxiv:2506.21328)**: 
   - Gini coefficient: 0.035
   - Min-max expert load ratio: 0.70
   - ⚠️ CV, Orthogonality, Overlap은 보고되지 않음

2. **Switch Transformer (Fedus et al., 2021)**: 
   - Balanced utilization: 94.8%
   - ⚠️ CV, Orthogonality, Overlap은 보고되지 않음

3. **Expert Choice (Zhou et al., 2022)**: 
   - Training convergence: 2× faster
   - ⚠️ CV, Orthogonality, Overlap은 보고되지 않음

4. **ERMoE (arxiv:2511.10971)**: 
   - "Natural flatter expert load distributions" (정량적 수치 없음)

5. **Advancing Expert Specialization (arxiv:2505.22323)**: 
   - Up to 23.79% performance gain

### 권장 접근 방법
1. **직접 측정**: 동일한 baseline에서 직접 측정하여 비교
2. **논문 재분석**: 논문의 figure, table, appendix에서 추출 가능한 정보 확인
3. **공개 코드**: GitHub repository에서 metrics 계산 코드 확인
4. **논문 저자 문의**: Metrics 데이터 요청 (가능한 경우)

### 비교 기준 재설정
- **LPR의 Gini 0.035**: Near-perfect balancing의 기준
- **Switch Transformer의 94.8% utilization**: 참고용
- **자체 측정값**: SPECTRA의 실제 측정값과 비교
- **Performance gain**: Advancing Expert Specialization의 23.79% gain과 비교

---

## ⚠️ 주의사항

### Routing Metrics 측정 시
1. **동일한 Base Model**: 모든 routing method는 동일한 base model 사용
2. **동일한 Expert Architecture**: Expert 구조는 동일하게 유지
3. **동일한 Training Setup**: Learning rate, batch size 등 모든 hyperparameter 동일
4. **Multiple Runs**: Routing metrics도 통계적 유의성 확보 (multiple seeds)
5. **Checkpoint 일관성**: 동일한 training step에서 비교

### Task Performance 측정 시
1. **Shot 수 일관성**: 모든 레퍼런스와 동일한 shot 수 사용
2. **평가 프레임워크**: lm-evaluation-harness 사용 권장 (표준화)
3. **데이터셋 버전**: 동일한 데이터셋 버전 사용

### 논문 작성 시
1. **Routing Metrics 우선**: Task performance보다 routing metrics를 먼저 제시
2. **인과관계 명확히**: Routing quality → Task performance 연결 설명
3. **Ablation Study 강조**: 각 component의 routing metrics 기여도 명시
4. **실제 측정값만 보고**: 추정값이나 확인되지 않은 값은 사용하지 않음

---

## 🔄 업데이트 로그

- 2025-11-28: 통합 문서 생성, 실제 논문 기반으로 재정리
- 향후 실험 결과에 따라 지속 업데이트 예정

---

## 📝 현재 SPECTRA 상태

### 측정 완료
- ✅ **Gram Orthogonality**: 0.94 (목표 달성)

### 측정 필요
- ⚠️ **Expert Overlap**: 측정 필요 (목표: < 15%)
- ⚠️ **Expert Entropy**: 측정 필요 (목표: ≥ 2.7)
- ⚠️ **Routing Consistency**: 측정 필요 (목표: > 85%)
- ⚠️ **Sequential Consistency**: 측정 필요 (목표: > 45%)

### 개선 필요
- ❌ **Load Balance CV**: 0.3 (목표: < 0.1, LPR Gini 0.035 기준)
- ⚠️ **Gini Coefficient**: 측정 필요 (목표: < 0.05, LPR: 0.035)
- ⚠️ **Min-max Expert Load Ratio**: 측정 필요 (목표: > 0.70, LPR: 0.70)

---

## 📊 벤치마크 데이터 소스

- **MMLU**: https://github.com/hendrycks/test
- **HellaSwag**: https://github.com/rowanz/hellaswag
- **GSM8K**: https://github.com/openai/grade-school-math
- **HumanEval**: https://github.com/openai/human-eval
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness

---

## 🔗 공개 모델 체크포인트

- **Mixtral-8x7B**: HuggingFace `mistralai/Mixtral-8x7B-v0.1`
- **LLaMA-2-7B**: HuggingFace `meta-llama/Llama-2-7b-hf`
- **GPT-2-Medium**: HuggingFace `gpt2-medium`
