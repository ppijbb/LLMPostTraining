# Seqorth MoE 논문 작성 가이드라인

## 1. 핵심 차별화 포인트 (Key Differentiators)

### 1.1 Gram Matrix 기반 Orthogonal Constraint
**주장**: Gram matrix를 활용한 orthogonal constraint가 expert diversity를 이론적으로 보장한다.

**이론적 배경**:
- Gram matrix `G = R @ R^T`는 expert representation의 내적 공간을 나타냄
- Identity matrix에 가까워질수록 expert들이 orthogonal한 표현 공간을 차지
- Gram-Schmidt 과정과의 연결: Gram matrix는 직교화 과정의 수학적 표현

**실험적 검증**:
- Gram matrix의 orthogonality score 측정
- Expert similarity matrix와의 상관관계 분석
- Ablation study: Gram matrix penalty 제거 시 성능 저하

### 1.2 Orthogonal Expression Projection
**주장**: Orthogonal projection을 통한 expression space 분리가 expert specialization을 촉진한다.

**차별화**:
- 기존 연구: 단순 linear projection 또는 attention-based routing
- 본 연구: Orthogonal constraint를 명시적으로 적용한 projection matrix
- Newton-Schulz iteration 또는 QR decomposition을 통한 orthogonalization

**검증**:
- Expression projection의 orthogonality 지표
- Expression-routing complementarity 분석
- Expert별 activation pattern의 다양성

### 1.3 Sequential Context-Aware Routing (GRU)
**주장**: Sequential routing이 context-dependent expert selection을 가능하게 한다.

**차별화**:
- 기존 연구: Token-level independent routing (Switch Transformer, GShard)
- 본 연구: GRU 기반 sequential hidden state로 이전 context 활용
- Global routing hidden state가 layer 간 정보 전달

**검증**:
- Sequential routing consistency 측정
- Context length에 따른 성능 변화
- Ablation study: GRU 제거 시 성능 저하

### 1.4 Domain Scoring: Cosine Similarity + Speciality Penalty
**주장**: Domain score (cosine similarity × speciality penalty)가 효과적인 routing decision을 만든다.

**차별화**:
- Cosine similarity: Expression과 routing logits 간의 유사도
- Speciality penalty: Gram matrix 기반 orthogonal constraint
- 결합: `domain_score = cosine_similarity × (1 + speciality_penalty)`

**검증**:
- Domain score와 실제 routing quality의 상관관계
- 각 component의 기여도 분석 (ablation)

### 1.5 Universal Upcycling Framework
**주장**: Any pretrained model을 MoE로 변환할 수 있는 범용 프레임워크 제공.

**차별화**:
- Model-agnostic layer discovery
- Dynamic signature detection (forward return values)
- 기존 MoE 모델의 router만 교체 가능

**검증**:
- 다양한 모델 (GPT-2, LLaMA, Qwen, Gemma, GPT-OSS)에서의 동작
- Upcycling 후 성능 유지/향상

## 2. 논문 구조 제안

### 2.1 Introduction
- **Problem**: MoE routing에서 expert diversity와 specialization의 trade-off
- **Contribution**: 
  1. Gram matrix 기반 orthogonal constraint
  2. Orthogonal expression projection
  3. Sequential context-aware routing
  4. Universal upcycling framework

### 2.2 Related Work
- **MoE Routing**: Switch Transformer, GShard, BASE Layers, Expert Choice
- **Orthogonal Constraints**: Orthogonal regularization, Gram matrix in neural networks
- **Sequential Routing**: GRU/RNN based routing (드물지만 존재)
- **Model Upcycling**: Dense-to-MoE conversion

### 2.3 Method
- **3.1 Gram Matrix-based Orthogonal Constraint**
  - Gram matrix 정의 및 orthogonal loss
  - Speciality penalty derivation
- **3.2 Orthogonal Expression Projection**
  - Expression projector 구조
  - Orthogonalization methods (Newton-Schulz, QR, SVD)
- **3.3 Sequential Routing with GRU**
  - GRU-based load balancer
  - Global routing hidden state
- **3.4 Domain Scoring**
  - Cosine similarity computation
  - Speciality penalty integration
- **3.5 Universal Upcycling Framework**
  - Layer discovery algorithm
  - Dynamic signature adaptation

### 2.4 Experiments
- **4.1 Experimental Setup**
  - Models: GPT-2, LLaMA, Qwen, Gemma, GPT-OSS
  - Tasks: Language modeling, downstream tasks
  - Baselines: Switch Transformer, GShard, Expert Choice
- **4.2 Main Results**
  - Performance comparison
  - Efficiency analysis
- **4.3 Ablation Studies**
  - Gram matrix penalty contribution
  - Expression projection contribution
  - Sequential routing contribution
  - Domain scoring components
- **4.4 Expert Specialization Analysis**
  - Expert activation patterns
  - Expert output diversity
  - Gram matrix quality metrics
- **4.5 Upcycling Analysis**
  - 다양한 모델에서의 upcycling 성공률
  - Upcycling 후 성능 변화

### 2.5 Discussion
- Gram matrix의 이론적 의미
- Sequential routing의 한계와 개선 방향
- Universal framework의 확장 가능성

## 3. 검증해야 할 주요 포인트

### 3.1 Expert Specialization
✅ **검증 방법**:
- Expert activation pattern 분석
- Expert output diversity 측정
- Expert similarity matrix 분석
- Expert별 dominant input pattern 발견

### 3.2 Gram Matrix Effectiveness
✅ **검증 방법**:
- Gram matrix orthogonality score
- Gram matrix penalty 제거 시 성능 변화
- Expert similarity와의 상관관계

### 3.3 Expression Projection Contribution
✅ **검증 방법**:
- Expression projection orthogonality
- Expression-routing complementarity
- Ablation: Expression projection 제거

### 3.4 Sequential Routing Benefit
✅ **검증 방법**:
- Sequential routing consistency
- Context length 실험
- Ablation: GRU 제거 (static routing)

### 3.5 Domain Scoring Quality
✅ **검증 방법**:
- Domain score와 routing quality 상관관계
- Cosine similarity vs. speciality penalty 기여도
- Ablation: 각 component 제거

### 3.6 Computational Efficiency
✅ **검증 방법**:
- Forward pass overhead 측정
- Memory usage 비교
- Training speed 비교

## 4. 추가 실험 아이디어

### 4.1 Expert Capacity Analysis
- Expert capacity에 따른 성능 변화
- Dynamic capacity allocation 가능성

### 4.2 Layer-wise Analysis
- Early layers vs. late layers의 routing 패턴
- Layer depth에 따른 specialization 정도

### 4.3 Task-specific Specialization
- Downstream task에서의 expert specialization
- Task transfer learning에서의 효과

### 4.4 Scaling Laws
- Expert 수에 따른 성능 scaling
- Model size에 따른 효과

## 5. 논문 제출 전 체크리스트

### 5.1 실험 완성도
- [ ] 모든 ablation study 완료
- [ ] 다양한 모델에서의 검증
- [ ] 다양한 task에서의 검증
- [ ] 통계적 유의성 검증

### 5.2 이론적 정당화
- [ ] Gram matrix의 수학적 배경 설명
- [ ] Orthogonal projection의 이론적 근거
- [ ] Sequential routing의 이론적 장점

### 5.3 코드 및 재현성
- [ ] 코드 공개 준비
- [ ] 재현 가능한 실험 설정
- [ ] Hyperparameter sensitivity 분석

### 5.4 논문 작성
- [ ] 명확한 contribution statement
- [ ] Related work와의 차별화 명확화
- [ ] 실험 결과의 해석
- [ ] 한계점 및 향후 연구 방향

## 6. 탑티어 컨퍼런스 제출을 위한 추가 포인트

### 6.1 Novelty 강조
- **Gram matrix를 MoE routing에 적용한 첫 번째 연구** (검증 필요)
- **Orthogonal expression projection의 명시적 활용**
- **Sequential routing의 효과적 결합**

### 6.2 실용성 강조
- **Universal upcycling framework**: 실제 적용 가능성
- **기존 모델과의 호환성**: 실무 적용 용이성
- **효율성**: 계산 오버헤드 최소화

### 6.3 이론적 깊이
- **Gram matrix의 이론적 의미 깊이 있게 설명**
- **Orthogonal constraint의 수학적 근거**
- **Sequential routing의 정보 이론적 해석**

### 6.4 실험적 엄밀성
- **통계적 유의성 검증**
- **다양한 baseline과의 비교**
- **Ablation study의 체계성**
- **Long-term training stability**

### 6.5 사회적 영향
- **Resource-efficient AI**: 적은 파라미터로 높은 성능
- **Democratization**: 더 작은 모델로도 MoE 활용 가능
- **Sustainability**: 효율적인 모델로 에너지 절약

