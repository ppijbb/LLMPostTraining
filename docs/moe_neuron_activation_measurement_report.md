# MoE 모델 Inference 시 활성화되는 Neuron 수 측정 방법 리포팅

## 개요

MoE(Mixture of Experts) 모델에서 inference 시 활성화되는 neuron 수를 측정하는 것은 모델의 효율성과 성능 평가에 핵심적입니다. 본 문서는 다양한 MoE 모델들의 구현 방식을 분석하고 측정 방법론을 정리합니다.

## 주요 MoE 모델들의 측정 방법

### 1. Expert 레벨 활성화 측정

#### 1.1 Switch Transformer / Mixtral 방식
- **측정 대상**: 활성화된 Expert 수 (Expert-level activation)
- **측정 방법**:
  - Router의 `top_k` 선택 결과를 기반으로 활성화된 Expert 수 계산
  - `selected_experts` 텐서에서 unique expert ID 추출
  - 각 레이어별로 활성화된 Expert 카운트

```python
# 예시 구현 패턴
active_experts = torch.unique(selected_experts).numel()
expert_usage_counts = torch.bincount(selected_experts.flatten(), minlength=num_experts)
active_experts_count = (expert_usage_counts > 0).sum().item()
```

#### 1.2 GShard / Expert Choice 방식
- **측정 대상**: Expert별 토큰 할당량 및 가중치 합
- **측정 방법**:
  - Expert별로 처리된 토큰 수 카운트
  - Routing weight의 합을 통한 활성화 강도 측정
  - Batch 및 sequence length에 따른 정규화

```python
# Expert별 weighted activation
expert_activation_counts = torch.zeros(num_experts)
expert_activation_weights = torch.zeros(num_experts)

for expert_idx in range(num_experts):
    mask = (selected_experts == expert_idx)
    expert_activation_counts[expert_idx] = mask.sum().float()
    expert_activation_weights[expert_idx] = routing_weights[mask].sum()
```

### 2. Neuron 레벨 활성화 측정

#### 2.1 FFN 내부 활성화 측정
- **목적**: Expert 내부의 실제 활성화된 뉴런 수 측정
- **방법**:
  - FFN(Feed-Forward Network)의 중간 활성화 값 추적
  - ReLU/GELU 활성화 후 non-zero 뉴런 카운트
  - Sparsity 측정을 통한 실제 연산량 추정

```python
# FFN 내부 활성화 추적
def count_activated_neurons(ffn_output, threshold=1e-6):
    # FFN의 중간 레이어 출력에서 활성화된 뉴런 수 계산
    activated = (ffn_output.abs() > threshold)
    return activated.sum(dim=-1)  # 각 토큰별 활성화된 뉴런 수
```

#### 2.2 Sparse Activation Tracking
- **목적**: 실제로 계산에 사용되는 뉴런만 추적
- **방법**:
  - Forward hook을 사용하여 각 레이어의 활성화 패턴 캡처
  - Expert별로 활성화된 뉴런의 비율 계산
  - Sparsity ratio = (활성화된 뉴런) / (전체 뉴런)

### 3. 실제 모델 사례

#### 3.1 Qwen3 (Alibaba)
- **전체 파라미터**: 2,350억 개
- **Inference 시 활성화**: 약 220억 개 (약 9.4%)
- **측정 방법**: 
  - Top-k expert selection (k=2 또는 k=4)
  - Expert별 활성화 빈도 모니터링
  - Layer-wise activation tracking

#### 3.2 DeepSeek-V3
- **측정 지표**:
  - Expert utilization rate
  - Layer별 활성화 분포
  - Routing entropy를 통한 활성화 다양성 측정

#### 3.3 MiniMax-01
- **전체 파라미터**: 4,560억 개
- **Inference 시 활성화**: 약 459억 개 (약 10%)
- **특징**: 하이브리드 어텐션과 MoE 아키텍처 결합

#### 3.4 Kanana-1.5 (Kakao)
- **전체 파라미터**: 157억 개
- **Inference 시 활성화**: 약 30억 개 (약 19%)
- **측정 방법**: 경량 멀티모달 구조와 결합된 MoE 활성화 추적

### 4. 현재 코드베이스의 구현

#### 4.1 현재 구현 상태 (`eval/moe_monitoring_callback.py`)

```python
# Expert usage tracking
usage_counts = torch.bincount(expert_assignments.long(), minlength=num_experts)
active_experts = (usage_counts > 0).sum().item()
unused_experts = (usage_counts == 0).sum().item()

# Metrics
layer_metrics.update({
    'usage_counts': usage_counts,
    'usage_distribution': usage_distribution,
    'active_experts': active_experts,
    'unused_experts': unused_experts,
})
```

#### 4.2 Seqorth 분석기 구현 (`eval/seqorth_analysis.py`)

```python
# Expert별 weighted activation count
expert_activation_counts = torch.zeros(self.num_experts, device=selected_experts.device)
expert_activation_weights = torch.zeros(self.num_experts, device=selected_experts.device)

for expert_idx in range(self.num_experts):
    mask = (selected_flat == expert_idx)
    expert_activation_counts[expert_idx] = mask.sum().float()
    expert_activation_weights[expert_idx] = weights_flat[mask].sum()

# Utilization rate
expert_utilization_rate = (expert_activation_counts > 0).float().mean().item()
```

### 5. 표준 측정 메트릭

#### 5.1 Expert 레벨 메트릭
1. **Active Experts Count**: 활성화된 Expert 수
2. **Expert Utilization Rate**: 활성화된 Expert 비율
3. **Expert Load Balance**: Expert 간 작업 분배 균형도
4. **Routing Entropy**: 라우팅 결정의 다양성

#### 5.2 Neuron 레벨 메트릭
1. **Activated Neurons per Token**: 토큰당 활성화된 뉴런 수
2. **Layer Sparsity**: 레이어별 활성화 sparsity
3. **Expert Internal Sparsity**: Expert 내부 뉴런 활성화 비율
4. **Computation Ratio**: 실제 연산량 / 전체 연산량

#### 5.3 통합 메트릭
1. **Effective Parameters**: Inference 시 실제 사용되는 파라미터 수
2. **Activation Ratio**: (활성화된 파라미터) / (전체 파라미터)
3. **FLOPs Efficiency**: 실제 FLOPs / 이론적 최대 FLOPs

### 6. 구현 패턴 비교

#### 6.1 Hook 기반 추적 (Transformers 라이브러리 스타일)
```python
def expert_activation_hook(module, input, output):
    # output에서 selected_experts 추출
    selected_experts = output[1] if isinstance(output, tuple) else None
    if selected_experts is not None:
        active_count = torch.unique(selected_experts).numel()
        return active_count

# Hook 등록
layer.register_forward_hook(expert_activation_hook)
```

#### 6.2 Forward Pass 내부 추적 (현재 코드베이스 스타일)
```python
# Forward pass 중 직접 카운트
expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts)
active_experts_per_layer = expert_mask.sum(dim=[0, 1]) > 0  # [num_experts]
active_count = active_experts_per_layer.sum().item()
```

#### 6.3 Callback 기반 모니터링 (현재 코드베이스)
```python
# Callback에서 메트릭 수집
def on_step_end(self, routing_info):
    expert_assignments = routing_info.get('expert_assignments')
    usage_counts = torch.bincount(expert_assignments, minlength=num_experts)
    self.expert_usage_history.append(usage_counts)
```

### 7. 추천 구현 방향

#### 7.1 현재 모델에 추가할 메트릭
1. **Neuron-level Activation Tracking**:
   - FFN 내부 활성화 추적
   - Expert별 실제 활성화된 뉴런 수 측정
   - Sparsity ratio 계산

2. **Layer-wise Aggregation**:
   - 레이어별 활성화 통계
   - 전체 모델의 활성화 파이프라인 추적

3. **Real-time Monitoring**:
   - Inference 중 실시간 활성화 패턴 로깅
   - 배치별/시퀀스별 활성화 변동성 추적

#### 7.2 구현 예시
```python
class NeuronActivationTracker:
    def __init__(self, num_experts, expert_dim):
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.activation_counts = defaultdict(list)
        
    def track_expert_internal_activation(self, expert_idx, ffn_output):
        # FFN 출력에서 활성화된 뉴런 수 계산
        activated = (ffn_output.abs() > 1e-6)
        activated_count = activated.sum(dim=-1)  # [batch*seq]
        self.activation_counts[expert_idx].append(activated_count)
        
    def get_activation_stats(self):
        stats = {}
        for expert_idx, counts in self.activation_counts.items():
            all_counts = torch.cat(counts)
            stats[f'expert_{expert_idx}'] = {
                'mean_activated_neurons': all_counts.float().mean().item(),
                'std_activated_neurons': all_counts.float().std().item(),
                'sparsity_ratio': 1.0 - (all_counts > 0).float().mean().item(),
            }
        return stats
```

### 8. 벤치마크 및 비교

#### 8.1 주요 모델 비교표

| 모델 | 전체 파라미터 | 활성화 파라미터 | 활성화 비율 | 측정 방법 |
|------|-------------|---------------|-----------|----------|
| Qwen3 | 235B | 22B | 9.4% | Expert-level + Layer-wise |
| MiniMax-01 | 456B | 45.9B | 10% | Expert routing + Attention hybrid |
| Kanana-1.5 | 15.7B | 3B | 19% | Lightweight MoE tracking |
| DeepSeek-V3 | - | - | - | Expert utilization + Routing entropy |

#### 8.2 측정 방법론 비교

| 측정 레벨 | Switch Transformer | Mixtral | GShard | 현재 코드베이스 |
|----------|-------------------|---------|--------|---------------|
| Expert Count | ✅ | ✅ | ✅ | ✅ |
| Expert Weights | ✅ | ✅ | ✅ | ✅ |
| Neuron Count | ❌ | ❌ | ⚠️ | ❌ |
| Layer Sparsity | ⚠️ | ⚠️ | ✅ | ⚠️ |
| Real-time Tracking | ⚠️ | ⚠️ | ✅ | ✅ |

### 9. 결론 및 권장사항

#### 9.1 현재 구현의 강점
- Expert 레벨 활성화 추적이 잘 구현되어 있음
- Callback 기반 모니터링으로 실시간 추적 가능
- Seqorth 분석기와 통합된 메트릭 수집

#### 9.2 개선 필요 사항
1. **Neuron-level Activation Tracking 추가**
   - FFN 내부 활성화 추적
   - Expert별 실제 활성화된 뉴런 수 측정

2. **Sparsity 메트릭 강화**
   - Layer-wise sparsity ratio
   - Expert 내부 sparsity 계산

3. **효율성 메트릭 추가**
   - Effective parameter count
   - Computation ratio (실제 FLOPs / 이론적 FLOPs)

4. **시각화 도구**
   - 레이어별 활성화 히트맵
   - Expert utilization 시계열 그래프

#### 9.3 Production-ready 측정 시스템
- Inference 중 실시간 모니터링
- 배치/시퀀스별 활성화 통계
- 로깅 및 메트릭 수집 자동화
- 성능 오버헤드 최소화

---

## 참고 자료

1. Switch Transformers: Scaling to Trillion Parameter Models
2. Mixtral of Experts (Mistral AI)
3. GShard: Scaling Giant Models with Efficient Conditional Computation
4. Qwen3 Technical Report (Alibaba Cloud)
5. DeepSeek-V3 Technical Documentation

