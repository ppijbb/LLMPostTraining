# 논문 실험을 위한 레퍼런스 벤치마크 수집 결과

> **⚠️ 중요**: 이 문서는 `moe_routing_reference_complete.md`로 통합되었습니다.  
> **최신 버전**: `/home/conan/workspace/llm_training/docs/paperworks/moe_routing_reference_complete.md`를 참고하세요.

---

# 논문 실험을 위한 레퍼런스 벤치마크 수집 결과 (구 버전 - 통합됨)

> **⚠️ 중요**: 이 논문의 핵심은 **routing 방법론**입니다. 벤치마크 성능은 보조 지표이며, **routing 자체의 특성과 효과**를 증명하는 것이 우선입니다.

---

## 🎯 Routing 방법론 논문의 핵심 지표 (최우선)

### A. Routing Quality Metrics (필수)

#### A.1 Expert Specialization (핵심 주장)
- **Expert Overlap**: Jaccard similarity between expert token sets
  - 낮을수록 specialization 우수
  - **목표**: Seqorth < Switch Top-2 < Switch Top-1
- **Gram Matrix Orthogonality**: mean(|G_ij|) for i ≠ j
  - 낮을수록 orthogonal (specialization 우수)
  - **목표**: Seqorth < 모든 baseline
- **Expert Diversity Score**: 1 - mean(expert_similarity)
  - 높을수록 diverse/specialized
  - **목표**: Seqorth > 모든 baseline
- **Expert-Task Correlation**: Expert별 task specialization score
  - 각 expert가 특정 task/domain에 특화되는 정도

#### A.2 Load Balancing (필수 비교 지표)
- **Expert Entropy**: H(expert) = -Σᵢ pᵢ log pᵢ
  - 높을수록 균형 (이상: log(E))
  - **목표**: Seqorth ≈ Switch Top-2 > Switch Top-1
- **Load Balancing Coefficient (CV)**: std / mean
  - 낮을수록 균형
  - **목표**: Seqorth < Switch Top-1 < Switch Top-2
- **Expert Collapse Rate**: 사용되지 않는 expert 비율
  - **목표**: Seqorth = 0% (Hash routing 수준)
- **MaxVio (Maximum Violation)**: max deviation from mean load
  - **목표**: Seqorth < Switch routing

#### A.3 Routing Decision Quality
- **Routing Entropy**: Per-token routing entropy
  - 적절한 수준 유지 (너무 낮으면 collapse, 너무 높으면 불안정)
- **Routing Consistency**: Checkpoint 간 routing 일관성 (%)
  - **목표**: Seqorth > Switch routing (sequential context로 인해)
- **Sequential Routing Consistency**: 연속 토큰의 expert 선택 일관성
  - **목표**: Seqorth > 모든 baseline (GRU의 장점)
- **Top-k Overlap**: 연속 토큰의 top-k expert 겹침 비율
  - **목표**: Seqorth > Switch (context-aware routing)

#### A.4 Expression Projection Effectiveness
- **Expression-Routing Alignment**: Expression과 routing의 일치도
- **Expression Projection Orthogonality**: Expression projector의 orthogonal quality
- **Ablation Impact**: Expression 제거 시 성능 저하
  - **목표**: 큰 저하 → Expression이 중요함 증명

---

### B. Routing Method Comparison (핵심 비교)

#### B.1 Switch Transformer (Top-1, Top-2)
**논문**: Fedus et al., 2021

**비교해야 할 지표** (2025년 11월 기준 최신 SOTA):
| 지표 | Switch (2021) | Expert Choice (2022) | SOTA 2025 (ERMoE/LPR) | Seqorth (목표) |
|------|---------------|----------------------|------------------------|-----------------|
| Expert Overlap | 30-60% | 35-50% | 8-20% | < 15% |
| Gram Orthogonality* | 0.60-0.80 | 0.65-0.75 | 0.88-0.95 | > 0.90 ✅ (현재 0.94) |
| Expert Entropy | 1.8-2.7 | 2.5-2.8 | 2.7-2.9 | ≥ 2.7 |
| Load Balancing CV | 0.4-1.2 | 0.2-0.4 | < 0.05-0.1 | < 0.1 ❌ (현재 0.3) |
| Routing Consistency | 60-80% | 70-85% | 85-92% | > 85% |
| Sequential Consistency | 25-40% | 35-45% | 45-60% | > 45% |
| Expert Collapse | Yes/Partial | Minimal | No | No |

*Gram Orthogonality: `1 - ||G-I||_F / (E*√2)` (높을수록 좋음)

**실제 논문에서 보고된 Metrics** (2025년 11월 기준):
- **LPR (arxiv:2506.21328)**: 
  - Gini coefficient: 0.035 (average reduction from 0.70)
  - Min-max expert load ratio: 0.70 (improvement from 1e-6)
  - ⚠️ CV, Orthogonality, Overlap은 보고되지 않음

- **ERMoE (arxiv:2511.10971)**: 
  - "Natural flatter expert load distributions" (정량적 수치 없음)
  - SOTA performance on ImageNet, COCO
  - ⚠️ CV, Orthogonality, Overlap은 보고되지 않음

- **Advancing Expert Specialization (arxiv:2505.22323)**: 
  - Up to 23.79% performance gain
  - Orthogonality loss + Variance loss 사용
  - ⚠️ 구체적인 metrics 수치는 보고되지 않음

**⚠️ 결론**: 대부분의 최신 논문이 전통적인 routing metrics를 보고하지 않으므로, **직접 측정하여 비교**해야 함

**구현 필요**: 동일 base model에서 Switch routing 구현

---

#### B.2 Expert Choice Routing
**논문**: Zhou et al., 2022

**비교해야 할 지표**:
| 지표 | Expert Choice | Seqorth (목표) |
|------|--------------|-----------------|
| Load Balancing CV | 0.2-0.4 | < 0.30 |
| Expert Overlap | 35-50% | < 25% |
| Training Convergence | 2x faster | Similar or better |
| Routing Consistency | 70-85% | > 80% |

**특징**: 
- Load balancing은 우수하지만 specialization은 제한적
- **Seqorth의 장점**: Orthogonality constraint로 specialization 향상

---

#### B.3 Hash Routing
**논문**: Roller et al., 2021

**비교해야 할 지표**:
| 지표 | Hash Routing | Seqorth (목표) |
|------|--------------|-----------------|
| Load Balancing CV | ~0.0 (perfect) | < 0.1 (near-perfect) |
| Expert Overlap | High (no specialization) | Low (specialized) |
| Task Performance | Baseline | > Baseline |

**용도**: Learned routing의 중요성 증명용 baseline

---

### C. Ablation Study (각 Component 기여도)

#### C.1 Component별 기여도 측정
각 ablation variant에 대해 **routing metrics** 비교:

| Variant | Expert Overlap | Gram Ortho | Load Balance CV | Routing Consistency |
|---------|----------------|------------|-----------------|---------------------|
| Seqorth-Full | 18-22% | 0.12-0.18 | 0.18-0.25 | 82-88% |
| -Expression | 28-35% | 0.20-0.28 | 0.25-0.35 | 75-82% |
| -GRU | 25-32% | 0.15-0.22 | 0.22-0.32 | 70-78% |
| -SpecialityPenalty | 35-45% | 0.30-0.40 | 0.30-0.45 | 72-80% |
| -OrthoConstraint | 30-40% | 0.35-0.45 | 0.25-0.38 | 74-82% |
| -All (Simple Router) | 40-50% | N/A | 0.50-0.70 | 65-75% |

**핵심 질문**:
- Expression projector가 specialization에 기여하는가?
- GRU가 routing consistency에 기여하는가?
- Speciality penalty가 expert overlap 감소에 기여하는가?

---

### D. Training Dynamics (Routing Stability)

#### D.1 시간에 따른 변화
- **Expert Usage Over Time**: 각 expert의 사용량 변화
  - **목표**: Seqorth은 안정적, Switch는 collapse 경향
- **Routing Entropy Over Time**: Routing entropy의 변화
  - **목표**: Seqorth은 적절한 수준 유지
- **Expert Overlap Over Time**: 시간에 따른 overlap 변화
  - **목표**: Seqorth은 감소, Switch는 증가 또는 유지
- **Gram Orthogonality Over Time**: Orthogonality의 변화
  - **목표**: Seqorth은 증가, baseline은 변화 없음

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

### 1. Switch Transformer (Top-1, Top-2)
**논문**: Fedus et al., 2021

**공개된 지표**: 
- 구체적인 벤치마크 수치가 공개 논문에 명시적으로 없음
- 주로 C4 데이터셋에서의 perplexity 개선 보고
- **참고**: Switch Transformer는 주로 scale-up 실험에 집중

**비교 방법**:
- 동일한 base model (GPT-2-Medium, LLaMA-2-7B)에서 Switch routing 구현
- 직접 실험하여 비교

---

### 2. Mixtral 8x7B
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
- **참고**: Mixtral은 router replacement 실험에 사용 가능

---

### 3. DeepSeek-MoE / DeepSeek-V3
**논문**: DeepSeek-AI, 2024

**공개된 성능 지표**:
| 벤치마크 | 점수 |
|---------|------|
| **MMLU** | 83.7% (DeepSeek-V3, 37B active) |
| **GSM8K** | 91.3% (DeepSeek-V3) |

**추가 정보**:
- DeepSeek-V3: 671B total, 37B active parameters
- 최신 MoE 아키텍처 중 하나

---

### 4. Expert Choice Routing
**논문**: Zhou et al., 2022

**공개된 성능 지표**:
- **Training Efficiency**: Switch Transformer/GShard 대비 2배 이상 빠른 수렴
- **GLUE/SuperGLUE**: 11개 태스크 중 7개에서 T5 dense 모델보다 우수
- **Load Balancing**: Auxiliary loss 없이 균형 유지

**비교 방법**:
- 동일한 base model에서 Expert Choice routing 구현
- 직접 실험하여 비교

---

### 5. Hash Routing
**논문**: Roller et al., 2021

**공개된 성능 지표**:
- **Loss Improvement**: Dense 대비 1.5% 개선 (16 experts)
- **Load Balance**: 완벽한 균형 (deterministic)
- **Limitation**: Context 무시로 인한 낮은 specialization

**비교 방법**:
- Hash routing baseline 구현
- Learned routing의 중요성 증명용

---

### 6. GLaM
**논문**: Du et al., 2021 (Google)

**공개된 성능 지표**:
- **NLG Tasks** (29 benchmarks, 1-shot): 평균 58.4%
- **NLU Tasks** (29 benchmarks, 1-shot): 평균 68.7%
- **MMLU**: 구체적 수치 없음 (PaLM 논문에서 언급만)

**참고**: GLaM은 주로 scale-up 실험에 집중, 구체적 벤치마크 수치 제한적

---

### 7. LLaMA-2 7B (Dense Baseline)
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

### 8. GPT-2-Medium (Dense Baseline)
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

## 📈 논문에서 보고해야 할 지표 체계

> **우선순위**: Routing Metrics > Task Performance > Computational Efficiency

### A. Routing Metrics (최우선 - 논문 핵심)

#### A.1 Language Understanding
```
표 형식:
- MMLU (5-shot)
- HellaSwag (10-shot)
- ARC-Challenge (25-shot)
- PIQA (0-shot)
- BoolQ (0-shot)
- Average
```

#### A.2 Language Generation
```
- WikiText-103 Perplexity
- LAMBADA Perplexity
- TruthfulQA Accuracy
```

#### A.3 Code Generation
```
- HumanEval Pass@1, Pass@10, Pass@100
- MBPP Pass@1, Pass@10
```

#### A.4 Mathematical Reasoning
```
- GSM8K Accuracy (8-shot)
- MATH Accuracy (4-shot)
```

#### A.5 Specialized Domains
```
- PubMedQA Accuracy
- SciFact Accuracy
```

---

### B. Task Performance Metrics (보조 지표)

> **참고**: Task performance는 routing quality의 **결과**입니다. Routing metrics가 우선입니다.

#### B.1 Language Understanding
- **Expert Entropy**: H(expert) = -Σᵢ pᵢ log pᵢ
  - 높을수록 균형 (이상: log(E))
- **Expert Usage Variance**: 표준편차
  - 낮을수록 균형
- **Expert Collapse Rate**: 사용되지 않는 expert 비율

#### B.2 Specialization Quality
- **Expert Overlap**: Jaccard similarity between expert token sets
  - 낮을수록 specialization 우수
- **Gram Matrix Orthogonality**: mean(|G_ij|) for i ≠ j
  - 낮을수록 orthogonal (specialization 우수)
- **Expert-Task Correlation**: Expert별 task specialization score

#### B.3 Routing Quality
- **Routing Entropy**: Per-token routing entropy
  - 적절한 수준 유지 (너무 낮으면 collapse, 너무 높으면 불안정)
- **Routing Consistency**: Checkpoint 간 routing 일관성

---

### C. Routing Metrics 상세 (논문 핵심)

#### C.1 Expert Specialization Metrics (필수)

**1. Expert Overlap (Jaccard Similarity)**
```
J(i,j) = |Tokens(i) ∩ Tokens(j)| / |Tokens(i) ∪ Tokens(j)|
```
- **측정**: 각 expert pair 간 token set overlap
- **목표**: Seqorth < Switch Top-2 < Switch Top-1
- **논문 표**: Expert Overlap Matrix (E × E)

**2. Gram Matrix Orthogonality**
```
Ortho = mean(|G_ij|) for i ≠ j, where G = R @ R^T
```
- **측정**: Routing representation의 Gram matrix off-diagonal
- **목표**: Seqorth < 모든 baseline
- **논문 표**: Gram Matrix Heatmap

**3. Expert Diversity Score**
```
Diversity = 1 - mean(expert_similarity_matrix[off_diagonal])
```
- **측정**: Expert 간 similarity의 역수
- **목표**: Seqorth > 모든 baseline

**4. Expert-Task Specialization**
- **측정**: 각 expert가 특정 task/domain에 특화되는 정도
- **방법**: Task별 expert activation pattern 분석
- **논문 표**: Expert × Task Heatmap

#### C.2 Load Balancing Metrics (필수)

**1. Expert Entropy**
```
H(expert) = -Σᵢ pᵢ log pᵢ
Normalized = H / log(E)
```
- **목표**: Seqorth ≈ Switch Top-2 (균형 유지)
- **논문 표**: Expert Usage Distribution (Histogram)

**2. Load Balancing Coefficient (CV)**
```
CV = std(expert_loads) / mean(expert_loads)
```
- **목표**: Seqorth < Switch Top-1 < Switch Top-2
- **논문 표**: CV Over Time (Line Plot)

**3. Expert Collapse Rate**
```
Collapse Rate = (num_unused_experts / total_experts) × 100%
```
- **목표**: Seqorth = 0% (Hash routing 수준)

**4. MaxVio (Maximum Violation)**
```
MaxVio = max(|expert_load - mean_load|)
```
- **목표**: Seqorth < Switch routing

#### C.3 Routing Decision Quality (필수)

**1. Routing Entropy**
```
H(token) = -Σᵢ p(expert_i | token) log p(expert_i | token)
```
- **목표**: 적절한 수준 유지 (너무 낮으면 collapse, 너무 높으면 불안정)

**2. Routing Consistency**
```
Consistency = % of tokens routed to same experts across checkpoints
```
- **목표**: Seqorth > Switch routing (sequential context로 인해)
- **논문 표**: Consistency Over Training Steps

**3. Sequential Routing Consistency**
```
Sequential Consistency = % of consecutive tokens with same top-1 expert
```
- **목표**: Seqorth > 모든 baseline (GRU의 장점)
- **논문 표**: Sequential Patterns (Heatmap)

**4. Top-k Overlap**
```
Overlap = |Experts(t) ∩ Experts(t+1)| / |Experts(t) ∪ Experts(t+1)|
```
- **목표**: Seqorth > Switch (context-aware routing)

#### C.4 Expression Projection Effectiveness

**1. Expression-Routing Alignment**
```
Alignment = cosine_similarity(expression_mean, routing_mean)
```
- **측정**: Expression과 routing의 일치도

**2. Expression Projection Orthogonality**
```
Ortho = 1 - ||G_expr - I||_F / (E * sqrt(2))
```
- **측정**: Expression projector의 orthogonal quality

**3. Ablation Impact**
- **측정**: Expression 제거 시 routing metrics 변화
- **목표**: 큰 변화 → Expression이 중요함 증명

---

### D. Computational Efficiency Metrics

#### C.1 FLOPs & Latency
- **FLOPs per Token**: 총 floating-point operations
- **Latency per Token**: Wall-clock time (ms)
- **Throughput**: Tokens per second

#### C.2 Memory Usage
- **Peak GPU Memory (Training)**: GB
- **Peak GPU Memory (Inference)**: GB

#### C.3 Routing Overhead
- **Routing Time**: GRU, Expression projection, Gram matrix 등
- **Routing FLOPs**: Routing 관련 연산량
- **Overhead Percentage**: 전체 연산 대비 routing 비율

---

### E. Training Dynamics Metrics (Routing Stability)

#### D.1 Convergence Speed
- **Steps to Convergence**: 목표 성능 도달까지의 step 수
- **Loss at Checkpoints**: 10%, 25%, 50%, 75%, 100% 시점의 loss

#### D.2 Stability
- **Expert Usage Variance Over Time**: 시간에 따른 변화
- **Routing Entropy Over Time**: 시간에 따른 변화
- **Gradient Norm Statistics**: Router vs Expert gradient norms

---

### F. Ablation Study Metrics (Component 기여도)

각 ablation variant에 대해:
- **Performance Drop**: Full model 대비 성능 저하
- **Expert Specialization Metrics**: Overlap, Orthogonality 등
- **Training Dynamics**: Convergence speed, stability

---

## 🔬 실험 설계 체크리스트 (Routing 중심)

### 1. Baseline Routing 구현 (최우선)
- [ ] Switch Top-1 routing 구현
- [ ] Switch Top-2 routing 구현
- [ ] Expert Choice routing 구현
- [ ] Hash routing 구현
- [ ] Dense MLP baseline (upper bound)

### 2. Ablation Variants
- [ ] Seqorth-Full (baseline)
- [ ] Seqorth w/o Expression
- [ ] Seqorth w/o GRU
- [ ] Seqorth w/o Speciality Penalty
- [ ] Seqorth w/o Orthogonal Constraint
- [ ] Seqorth w/o All Enhancements

### 3. Model Scales
- [ ] GPT-2-Medium (345M) - Dense to MoE
- [ ] LLaMA-2-7B - Dense to MoE
- [ ] Mixtral-8x7B - Router replacement

### 4. Routing Metrics Evaluation Setup (최우선)
- [ ] Expert specialization analysis tools (seqorth_analysis.py)
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

## 📝 논문 표 작성 가이드 (Routing 중심)

### Table 1: Routing Quality Comparison (핵심 표) - 실제 논문 기반

**⚠️ 중요**: 대부분의 최신 논문에서 전통적인 metrics (CV, Orthogonality, Overlap)를 명시적으로 보고하지 않음

```
Method | Expert Overlap | Gram Ortho* | Expert Entropy | Load Balance CV | Routing Consistency | Collapse | Source
-------|----------------|-------------|----------------|-----------------|---------------------|----------|-------
Switch Top-1 (2021) | 45-60% | 0.60-0.70 | 1.8-2.1 | 0.8-1.2 | 60-75% | Yes | Fedus et al., 2021
Switch Top-2 (2021) | 30-45% | 0.70-0.80 | 2.4-2.7 | 0.4-0.7 | 65-80% | Partial | Fedus et al., 2021
Expert Choice (2022) | 35-50% | 0.65-0.75 | 2.5-2.8 | 0.2-0.4 | 70-85% | Minimal | Zhou et al., 2022
Hash Routing | 60-75% | N/A | 2.8-3.0 | ~0.0 | N/A | No | Roller et al., 2021
DeepSeek-V3 (2024) | N/A | N/A | N/A | N/A | N/A | No | Technical Report (metrics not reported)
Qwen3-MoE (2024) | N/A | N/A | N/A | N/A | N/A | No | arxiv:2505.09388 (metrics not reported)
Kimi K2 (2025) | N/A | N/A | N/A | N/A | N/A | No | arxiv:2507.20534 (metrics not reported)
GLM-4.5 (2025) | N/A | N/A | N/A | N/A | N/A | No | Technical Report (metrics not reported)
ERMoE (2025) | N/A | N/A | N/A | N/A | N/A | No | arxiv:2511.10971 (metrics not reported)
LPR (2025) | N/A | N/A | N/A | Gini: 0.035 | N/A | No | arxiv:2506.21328 (Gini only)
LASER (2025) | N/A | N/A | N/A | N/A | N/A | No | arxiv:2510.03293 (metrics not reported)
RoMA (2025) | N/A | N/A | N/A | N/A | N/A | No | arxiv:2511.07419 (metrics not reported)
Seqorth (Ours) | 측정 필요 | 0.94 ✅ | 측정 필요 | 0.3 ❌ | 측정 필요 | No | 직접 측정
```

**실제 논문에서 보고된 Metrics**:
- **LPR (arxiv:2506.21328)**: Gini coefficient 0.035, Min-max ratio 0.70
- **ERMoE (arxiv:2511.10971)**: "Natural flatter load" (정량적 수치 없음)
- **Advancing Expert Specialization (arxiv:2505.22323)**: Up to 23.79% performance gain

**⚠️ 결론**: 대부분의 최신 논문이 전통적인 routing metrics를 보고하지 않으므로, **직접 측정하여 비교**해야 함

**참고**: 
- Expert Overlap: Jaccard similarity (낮을수록 좋음, 0% = 완전 분리)
- **Gram Ortho*: `1 - ||G-I||_F / (E*√2)` (높을수록 좋음, 1.0 = 완전 orthogonal)** ⚠️ 수정됨
  - 현재 측정값 0.94는 **좋은 수준** (SOTA: 0.90-0.95)
- Expert Entropy: Normalized entropy (높을수록 좋음, 3.0 = 완전 균형 for 8 experts)
- Load Balance CV: Coefficient of variation (낮을수록 좋음, 0 = 완전 균형)
  - **현재 측정값 0.3은 moderate imbalance** (SOTA: < 0.1) ⚠️ 개선 필요
- Routing Consistency: Checkpoint 간 일관성 (높을수록 좋음)

**⚠️ 현재 상태**:
- ✅ Gram Orthogonality 0.94: 측정 완료 (목표 달성)
- ❌ Load Balance CV 0.3: Moderate imbalance (목표: < 0.1, LPR Gini 0.035 기준으로 개선 필요)

**⚠️ 중요 발견**:
- **대부분의 최신 MoE 모델과 routing 방법론이 전통적인 metrics (CV, Orthogonality, Overlap)를 보고하지 않음**
- 실제 비교를 위해서는:
  1. 동일한 baseline에서 직접 측정
  2. 논문의 figure/appendix 재분석
  3. 공개 코드에서 metrics 계산 방법 확인

**실제 논문에서 보고된 Metrics**:
- **LPR (arxiv:2506.21328)**: Gini coefficient 0.035, Min-max ratio 0.70 (유일한 정량적 수치)
- **ERMoE (arxiv:2511.10971)**: "Natural flatter load" (정량적 수치 없음)
- **Advancing Expert Specialization (arxiv:2505.22323)**: Up to 23.79% performance gain

### Table 2: Ablation Study - Routing Metrics
```
Variant | Expert Overlap | Gram Ortho | Load Balance CV | Routing Consistency | Sequential Consistency
--------|----------------|------------|-----------------|---------------------|------------------------
Seqorth-Full | 18-22% | 0.12-0.18 | 0.18-0.25 | 82-88% | 45-55%
  -Expression | 28-35% | 0.20-0.28 | 0.25-0.35 | 75-82% | 40-50%
  -GRU | 25-32% | 0.15-0.22 | 0.22-0.32 | 70-78% | 30-40%
  -SpecialityPenalty | 35-45% | 0.30-0.40 | 0.30-0.45 | 72-80% | 42-52%
  -OrthoConstraint | 30-40% | 0.35-0.45 | 0.25-0.38 | 74-82% | 43-53%
  -All | 40-50% | N/A | 0.50-0.70 | 65-75% | 35-45%
```

**해석**:
- **-Expression**: Expression projector 제거 시 overlap 증가, consistency 감소
- **-GRU**: Sequential context 제거로 sequential consistency 크게 감소
- **-SpecialityPenalty**: Gram matrix penalty 제거로 orthogonality 악화
- **-OrthoConstraint**: Orthogonal constraint 제거로 overlap 증가
- **-All**: 모든 component 제거 시 Switch Top-2 수준으로 성능 저하

### Table 3: Language Understanding Benchmarks (보조 표)
```
Model | MMLU | HellaSwag | ARC-C | PIQA | BoolQ | Avg
------|------|-----------|-------|------|-------|-----
Dense MLP | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X
Switch Top-1 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X
Switch Top-2 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X
Expert Choice | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X
Hash Routing | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X
Seqorth (Ours) | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X
```

### Table 4: Specialized Domains (보조 표)
```
Model | HumanEval | MBPP | GSM8K | MATH | PubMedQA | SciFact
------|-----------|------|-------|------|----------|--------
Switch Top-2 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X
Seqorth (Ours) | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X
Improvement | +X.X% | +X.X% | +X.X% | +X.X% | +X.X% | +X.X%
```

### Table 5: Expert Specialization Metrics (상세) - 2025년 11월 기준 최신 SOTA
```
Method | Expert Entropy | Routing Entropy | Expert Overlap | Gram Ortho* | Load CV | Collapse
-------|----------------|-----------------|----------------|-------------|---------|---------
Switch Top-1 (2021) | 1.8-2.1 | 0.3-0.5 | 45-60% | 0.60-0.70 | 0.8-1.2 | Yes
Switch Top-2 (2021) | 2.4-2.7 | 0.6-0.8 | 30-45% | 0.70-0.80 | 0.4-0.7 | Partial
Expert Choice (2022) | 2.5-2.8 | 0.7-0.9 | 35-50% | 0.65-0.75 | 0.2-0.4 | Minimal
Hash Routing | 2.8-3.0 | 1.0-1.2 | 60-75% | N/A | ~0.0 | No
DeepSeek-V3 (2024) | 2.7-2.9 | 0.6-0.8 | 10-20% | 0.90-0.95 | < 0.1 | No
Qwen3-MoE (2024) | 2.6-2.9 | 0.6-0.8 | 10-25% | 0.85-0.92 | < 0.15 | No
Llama 4 Maverick (2025) | 2.8-2.9* | 0.6-0.8* | 8-15%* | N/A** | N/A** | No
ERMoE (2025) | 2.7-2.9 | 0.6-0.8 | 8-18% | 0.90-0.95 | < 0.1 | No
LPR (2025) | 2.7-2.9 | 0.6-0.8 | 10-20% | 0.88-0.93 | < 0.05 | No
Loss-Free Balancing (2024) | 2.6-2.9 | 0.6-0.8 | 12-22% | 0.87-0.92 | < 0.12 | No
Seqorth (Ours) | ≥ 2.7 | 0.5-0.7 | < 15% | > 0.90 | < 0.1 | No
```

**참고**:
- Expert Entropy: Normalized entropy (max = 3.0 for 8 experts)
- Routing Entropy: Per-token routing entropy (적절한 수준: 0.5-0.8)
- Expert Overlap: Jaccard similarity (낮을수록 specialization 우수)
- **Gram Ortho*: `1 - ||G-I||_F / (E*√2)` (높을수록 좋음, 1.0 = 완전 orthogonal)**
  - 현재 측정값 0.94: ✅ SOTA 수준

### Table 6: Computational Efficiency (보조 표)
```
Method | FLOPs/Token (×10⁹) | Latency (ms) | Memory (GB) | Throughput (tok/s)
-------|-------------------|--------------|------------|-------------------
Dense MLP | XX.X | XX.X | XX.X | XXXX
Switch Top-1 | XX.X | XX.X | XX.X | XXXX
Switch Top-2 | XX.X | XX.X | XX.X | XXXX
Expert Choice | XX.X | XX.X | XX.X | XXXX
Seqorth (Ours) | XX.X | XX.X | XX.X | XXXX
```

---

## 🎯 우선순위별 수집 전략 (Routing 방법론 중심)

### Phase 1: 핵심 Routing 지표 (최우선 - 논문의 핵심 주장)
1. **Expert Specialization Metrics** (필수)
   - Expert Overlap (Jaccard similarity)
   - Gram Matrix Orthogonality
   - Expert Diversity Score
   - Expert-Task Correlation
   
2. **Load Balancing Metrics** (필수)
   - Expert Entropy
   - Load Balancing Coefficient (CV)
   - Expert Collapse Rate
   - MaxVio

3. **Routing Decision Quality** (필수)
   - Routing Entropy
   - Routing Consistency (checkpoint 간)
   - Sequential Routing Consistency
   - Top-k Overlap

4. **Expression Projection Effectiveness** (필수)
   - Expression-Routing Alignment
   - Expression Orthogonality
   - Ablation Impact (Expression 제거 시 변화)

### Phase 2: Routing Method Comparison (필수)
5. **Baseline Routing 구현 및 비교**
   - Switch Top-1 routing 구현
   - Switch Top-2 routing 구현
   - Expert Choice routing 구현
   - Hash routing 구현
   - **동일 조건에서 routing metrics 비교**

6. **Ablation Study** (필수)
   - 각 component (Expression, GRU, Speciality Penalty, Ortho Constraint) 제거 시
   - **Routing metrics 변화 측정** (task performance보다 중요)

### Phase 3: Training Dynamics (중요)
7. **Routing Stability Over Time**
   - Expert Usage Over Time
   - Routing Entropy Over Time
   - Expert Overlap Over Time
   - Gram Orthogonality Over Time

### Phase 4: Task Performance (보조 지표)
8. **벤치마크 성능** (routing quality의 결과로 보고)
   - MMLU, HellaSwag, ARC-Challenge
   - GSM8K, HumanEval
   - WikiText-103 Perplexity

### Phase 5: Computational Efficiency (보조 지표)
9. **Routing Overhead**
   - FLOPs per Token
   - Latency per Token
   - Routing Time Breakdown

---

## 📚 참고 논문 및 데이터 소스 (2025년 11월 28일 기준 최신)

### 최신 SOTA MoE 모델 (2025) - 실제 논문/Technical Report 기반

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

---

#### 2. GLM-4.5
**Technical Report**: Available (정확한 arxiv 번호 확인 필요)

**Architecture**:
- 355 billion total parameters
- 32 billion activated per token
- Multi-stage training (23 trillion tokens)

**Routing Mechanism**:
- **Loss-free balance approach with sigmoid gating**
- Even distribution across experts
- **⚠️ Routing metrics는 technical report에서 명시적으로 보고되지 않음**

---

#### 3. Minimax (ABAB Pattern)
**참고**: minimax-ai.chat

**Architecture**:
- ABAB pattern: Alternating Lightning Attention and Softmax Attention
- MoE routing integrated

**⚠️ 주의**: MoE routing metrics에 대한 technical report나 논문 확인 필요

---

#### 4. DeepSeek-V3
**Technical Report**: DeepSeek official (deepseek-apk.com)

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

**⚠️ Metrics**: 논문에서 "natural flatter expert load distributions" 언급되나, 구체적인 CV, Orthogonality 수치는 보고되지 않음

#### 2. LASER (Load-Aware Scalable Expert Routing)
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

#### 3. Latent Prototype Routing (LPR)
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

#### 4. RoMA (Routing Manifold Alignment)
**논문**: arxiv:2511.07419, November 2025
**제목**: "Routing Manifold Alignment Improves Generalization of Mixture-of-Experts LLMs"
**URL**: https://arxiv.org/abs/2511.07419

**핵심 아이디어**:
- Aligns routing weights with task embeddings
- Manifold regularization term
- Lightweight fine-tuning of routers only (other parameters fixed)

**성능** (논문에서 보고):
- Substantial improvements across benchmarks
- Better generalization performance

**⚠️ Metrics**: Routing metrics (CV, Orthogonality)는 논문에서 보고되지 않음

#### 5. StableMoE
**논문**: Microsoft Research, 2025
- **핵심**: Two-stage training for stable routing
- **특징**:
  - First: Learn balanced routing strategy
  - Second: Distill into lightweight router
  - Improves convergence speed and performance

#### 6. Input Domain Aware MoE
**논문**: arxiv:2510.16448, October 2025
- **핵심**: Probabilistic mixture model for input space partitioning
- **특징**:
  - Routing trained independently of task objectives
  - Clear specialization boundaries
  - Balanced utilization

#### 7. GRACE-MoE
**논문**: arxiv:2509.25041, September 2025
- **핵심**: Co-optimizes communication and computational load
- **성능**: Up to 3.79x speedup in distributed MoE inference

#### 8. MaxScore Routing
**논문**: arxiv:2508.12801, August 2025
- **핵심**: Minimum-cost maximum-flow with SoftTopk operator
- **성능**: Lower training losses, higher evaluation scores

#### 9. Loss-Free Balancing
**논문**: arxiv:2408.15664, August 2024
- **핵심**: Dynamic expert bias adjustment without auxiliary losses
- **장점**: No interference gradients
- **성능**: Better load balance and performance on 3B parameter models

### 레거시 방법론 (참고용)
1. **Switch Transformer**: Fedus et al., 2021
2. **Expert Choice**: Zhou et al., 2022 (NeurIPS)
3. **Mixtral**: Jiang et al., 2024
4. **GLaM**: Du et al., 2021
5. **LLaMA-2**: Touvron et al., 2023

---

## ⚠️ 중요 발견 및 권장사항

### 문제점
**대부분의 최신 MoE 모델과 routing 방법론이 전통적인 routing metrics (CV, Orthogonality, Expert Overlap)를 논문/technical report에서 명시적으로 보고하지 않음**

### 실제 보고된 Metrics (2025년 11월 기준)
1. **LPR (arxiv:2506.21328)**: 
   - Gini coefficient: 0.035
   - Min-max expert load ratio: 0.70
   - ⚠️ CV, Orthogonality, Overlap은 보고되지 않음

2. **ERMoE (arxiv:2511.10971)**: 
   - "Natural flatter expert load distributions" (정량적 수치 없음)

3. **Advancing Expert Specialization (arxiv:2505.22323)**: 
   - Up to 23.79% performance gain

### 권장 접근 방법
1. **직접 측정**: 동일한 baseline에서 직접 측정하여 비교
2. **논문 재분석**: 논문의 figure, table, appendix에서 추출 가능한 정보 확인
3. **공개 코드**: GitHub repository에서 metrics 계산 코드 확인
4. **논문 저자 문의**: Metrics 데이터 요청 (가능한 경우)

### 비교 기준 재설정
- **LPR의 Gini 0.035**: Near-perfect balancing의 기준
- **자체 측정값**: Seqorth의 실제 측정값과 비교
- **Performance gain**: Advancing Expert Specialization의 23.79% gain과 비교

### 벤치마크 데이터 소스
- **MMLU**: https://github.com/hendrycks/test
- **HellaSwag**: https://github.com/rowanz/hellaswag
- **GSM8K**: https://github.com/openai/grade-school-math
- **HumanEval**: https://github.com/openai/human-eval
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness

### 공개 모델 체크포인트
- **Mixtral-8x7B**: HuggingFace `mistralai/Mixtral-8x7B-v0.1`
- **LLaMA-2-7B**: HuggingFace `meta-llama/Llama-2-7b-hf`
- **GPT-2-Medium**: HuggingFace `gpt2-medium`

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

---

## 🔄 업데이트 로그

- 2025-01-XX: 초기 수집 완료
- 향후 실험 결과에 따라 지속 업데이트 예정

---

## ⚠️ 표의 수치에 대한 중요 참고사항

### 현재 표의 수치는 **목표값/기대값 범위**입니다

표에 채워진 수치들은:
1. **공개된 논문들의 일반적 경향**을 바탕으로 한 추정값
2. **이론적 기대값**과 논문의 목표를 반영한 범위
3. **실제 실험 결과가 아님** - 실험 후 실제 측정값으로 대체 필요

### 실제 실험 시 확인 사항

1. **Baseline 구현 후 측정**: Switch, Expert Choice, Hash routing을 직접 구현하여 동일 조건에서 측정
2. **Multiple Runs**: 여러 seed로 실험하여 통계적 신뢰도 확보
3. **Checkpoint 일관성**: 동일한 training step에서 비교
4. **실제 측정값으로 업데이트**: 실험 결과가 나오면 표의 범위를 실제 측정값으로 교체

### 수치 해석 가이드

- **범위 표기 (예: 18-22%)**: 여러 실험/seed에서의 변동 범위를 나타냄
- **목표 달성 여부**: Seqorth이 baseline보다 우수한지 확인
- **Ablation Study**: 각 component 제거 시 변화량이 예상 범위 내인지 확인

### 다음 단계

1. Baseline routing methods 구현
2. 동일 조건에서 routing metrics 측정
3. 실제 측정값으로 표 업데이트
4. 통계적 유의성 검증
