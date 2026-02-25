## G3MoE: 라우팅 균형과 전문가 특화를 동시에 달성하는 차세대 Mixture-of-Experts

### 초록
Mixture-of-Experts(MoE)는 파라미터 효율성과 확장성에서 큰 이점을 제공하지만, 학습 중 라우팅 붕괴(routing collapse)와 전문가 불균형(expert imbalance) 문제가 빈번하다. 본 논문은 G3MoE(Generative 3rd MoE)라는 새로운 MoE 설계를 제안한다. 핵심 기여는 다음과 같다: (1) Switch Transformer 계열의 보조 손실(aux loss)과 z-loss에 더해, 라우팅 엔트로피 정규화와 평균 사용률 균등화(usage uniformity) 손실을 도입하여 라우팅 붕괴를 방지한다. (2) EMA 기반의 적응형 로짓 필터(사용량 패널티)와 전문가 특화(embedding) 추적을 통해 전문가 간 기능적 다양성과 부하 분산을 함께 강화한다. (3) 전문가 가중치의 직교화(orthogonalization) 손실을 추가하여 중복 기능 학습을 억제한다. (4) 실용적 학습 설정(SFT + DeepSpeed, BF16, 모니터링 콜백)을 제시하고, 불균형과 붕괴를 조기에 탐지·교정하는 모니터링 지표를 제공한다. 본 접근은 히트맵 상의 특정 전문가 쏠림을 완화하고, `moe/avg_routing_entropy` 증대와 `moe/avg_expert_cv` 감소를 일관되게 유도한다.

### 1. 서론
대규모 언어 모델에서 MoE는 추론 비용을 억제하면서 파라미터 규모를 확장하는 대표적인 방법이다. 그러나 실제 학습에서는 라우터가 소수 전문가에 과도하게 집중하거나, 확률 분포의 엔트로피가 붕괴하여 토큰이 거의 동일한 전문가로만 흘러가는 문제가 빈번하다. 이는 학습 불안정성, 특정 전문가 과적합, 성능 정체로 이어진다.

본 연구는 G3MoE라는 실전 지향의 설계를 통해 다음 목표를 달성한다: (i) 라우팅 분포의 다양성 유지, (ii) 전문가 부하의 균형, (iii) 전문가 간 기능적 분화, (iv) 학습 안정성. 이를 위해 기존 보조 손실을 강화하고, 라우터 신호에 직접 작용하는 정규화 항을 도입하며, EMA 기반 적응형 패널티와 특화 추적을 함께 사용한다.

### 2. 관련 연구
- Switch Transformer: 토큰 분배 균형을 위한 보조 손실과 z-loss 도입으로 라우터 과신을 억제.
- GShard/Sparse MoE: 대규모 전문가 병렬화와 드랍/패딩 전략.
- Entropy regularization: 확률 분포의 다양성 유지를 위한 일반적 기법.
본 연구는 이러한 계열의 장점을 결합하고, 실제 학습에서 관측되는 불균형/붕괴 시나리오에 특화된 정규화 항과 모니터링·튜닝 가이드라인을 제공한다.

### 3. G3MoE 아키텍처
#### 3.1 텍스트 백본과 전문가 레이어
G3MoE는 텍스트 백본 위에 라우트형 MLP 전문가를 배치한다. 각 전문가 MLP는 `gate_proj`, `up_proj`, `down_proj` 구조를 갖고, 토큰은 라우터에 의해 상위 k개의 전문가로 라우팅된다. 일부 레이어는 초기에 dense 대체(`first_k_dense_replace`)로 안정적 수렴을 돕는다.

#### 3.2 라우터와 스파스 믹서(SparseMixer-v2)
- 라우터는 선형층 출력 `router_logits`로부터 확률/스코어를 계산한다.
- 본 연구의 구현에서 사용하는 `sparsemixer`는 GRIN-MoE가 제안한 SparseMixer-v2 계열 기법을 따르며, top-2 선택과 Gumbel 기반 명시적 샘플링을 사용하고, Heun의 3차 방법을 이용한 그래디언트 근사로 이산 라우팅에 대한 안정적인 역전파를 제공한다. 이는 표준 TopK의 비미분성을 우회해 라우팅 파라미터에 유효한 그래디언트를 전달한다 [GRIN-MoE].
  - 핵심 아이디어: (i) TopK를 무작위 샘플링 기반의 선택으로 치환, (ii) 선택 확률의 소프트맥스 분포 위에서 Heun 계열의 추정기를 적용해 일관된 그래디언트 신호를 구성.
  - 구현 세부: 학습 모드에서 Gumbel-Softmax 유사 샘플링으로 전문가를 선택하고, Heun 계열의 중간점 보정 마스크를 적용하여 라우팅 스코어에 대한 미분 경로를 부여한다(코드 내 mp.apply 및 midpoint mask 로직 참조).
- 학습 중 `router_jitter_noise`, `input_jitter_noise`를 적용해 라우터 과신(over-confidence)을 추가로 완화한다.
- 그룹 기반 top-k 선택(`n_group`, `topk_group`)으로 구조적 다양성을 유지한다.

참고: 명칭이 유사한 “Sparse Mixer(2022)”는 BERT 인코더 효율화를 위한 별개의 모델로, 본 논문에서 사용하는 SparseMixer-v2(라우팅용 그래디언트 추정기)와는 다른 계열이다. 혼동을 피하기 위해 본문에서는 GRIN-MoE의 SparseMixer-v2로 명시한다 [SparseMixer-Enc]와 구분.

#### 3.3 EMA 기반 적응형 부하 패널티와 전문가 특화 추적
- 적응형 패널티: 스텝마다 전문가 사용량의 EMA(`expert_load_ema`)를 갱신하고, 과다 사용 전문가의 로짓을 미세하게 감산한다(패널티 강도 `balancing_strength`).
- 특화 추적: 전문가별 특화 임베딩의 EMA(`expert_specialization_ema`)를 유지하고, 입력 히든과의 정규화 코사인 유사도를 라우터 로짓에 보너스로 더해 특화를 촉진한다(`specialization_strength`).

#### 3.4 전문가 직교화(Orthogonalization)
전문가 `down_proj` 가중치들을 행 단위로 정규화한 행렬 V에 대해 \(\|VV^\top - I\|_F^2\)를 최소화하여 기능 중복을 억제한다. 계수는 `ortho_loss_coef`로 제어한다.

### 4. 손실 함수 설계
총 손실은 다음 항들의 가중 합이다.

\[\begin{aligned}
\mathcal{L} &= \mathcal{L}_{task} 
\; + \; \lambda_{lb} \; \mathcal{L}_{lb} 
\; + \; \lambda_{z} \; \mathcal{L}_{z}
\; + \; \lambda_{ent} \; \mathcal{L}_{ent}
\; + \; \lambda_{uni} \; \mathcal{L}_{uni}
\; + \; \lambda_{ortho} \; \mathcal{L}_{ortho} \;.
\end{aligned}\]

- \(\mathcal{L}_{task}\): 표준 언어모델링 손실(크로스 엔트로피).
- \(\mathcal{L}_{lb}\) (Switch aux): 라우팅 불균형을 억제하기 위한 보조 손실. 층/토큰 평균에서 전문가별 토큰 비율과 라우터 확률의 내적을 최소화한다. 구현상 마스크를 반영해 패딩 토큰의 영향을 제거한다.
- \(\mathcal{L}_{z}\) (z-loss): \(\mathrm{logsumexp}(\text{logits})\)의 제곱 평균을 최소화하여 라우터 과신을 억제한다.
- \(\mathcal{L}_{ent}\) (엔트로피 정규화): 토큰별 라우터 분포 엔트로피 \(H(p)\)의 음수를 최소화(=엔트로피 최대화)한다. 스케일 불변성을 위해 \(\log E\)로 정규화한다. 
  \[ \mathcal{L}_{ent} = -\,\mathbb{E}_{tokens}\left[ \tfrac{H(p)}{\log E} \right], \quad H(p)= -\sum_i p_i\log p_i. \]
- \(\mathcal{L}_{uni}\) (사용률 균등화): 평균 라우터 확률(전 토큰 평균)이 균등 분포에 가깝도록 \(\ell_2\) 손실을 적용한다.
  \[ \mathcal{L}_{uni} = \big\| \bar{p} - \tfrac{1}{E}\mathbf{1} \big\|_2^2. \]
- \(\mathcal{L}_{ortho}\): 전문가 가중치 직교화 손실(§3.4).

권장 초기 계수(텍스트 7B급 기준):
- \(\lambda_{lb}=\) `router_aux_loss_coef` ≈ 1e-3
- \(\lambda_{z}=\) `router_z_loss_coef` ≈ 1e-4 ~ 5e-4
- \(\lambda_{ent}=\) `router_entropy_coef` ≈ 1e-3 ~ 5e-3
- \(\lambda_{uni}=\) `usage_uniformity_coef` ≈ 1e-4 ~ 1e-3
- \(\lambda_{ortho}=\) `ortho_loss_coef` ≈ 1e-2

튜닝 가이드: 엔트로피가 낮고 쏠림이 심하면 `router_entropy_coef`/`usage_uniformity_coef`를 소폭 증가. 라우팅이 과도히 무작위화되면 두 계수를 낮춘다. `router_aux_loss_coef`는 균형 유지의 기본축으로 소폭 조정한다.

### 5. 학습 설정
- SFT + DeepSpeed, BF16, gradient checkpointing.
- 일부 초기 레이어는 dense 대체로 안정화(`first_k_dense_replace`).
- 라우터 파라미터(선형 라우터, `routing_temperature`)는 반드시 학습 가능 상태 유지.
- Jitter 노이즈(`router_jitter_noise`, `input_jitter_noise`)는 소량으로 시작.
- LoRA 사용 시, 라우터/온도는 풀파라미터 학습을 유지하여 라우팅 적응성을 보장.

### 6. 모니터링과 평가
훈련 중 콜백으로 다음 지표를 추적한다.
- `moe/avg_routing_entropy`: 라우팅 다양성(높을수록 양호)
- `moe/avg_expert_cv`: 전문가 사용량 변동계수(낮을수록 균등)
- `moe/total_unused_experts`: 미사용 전문가 수(낮을수록 양호)
- 레이어별 히트맵: 시간에 따른 전문가 사용률 분포 시각화
경고 기준(예): `max_usage_ratio` > 4.0, `routing_entropy` < 0.1, `unused_experts` 비율 > 25%.

벤치마크: MMLU, HellaSwag, GSM8K, TruthfulQA, ARC, PIQA 등. 

### 7. 실험 설계 및 관찰
- 아블레이션: (a) 기본 Switch aux + z-loss, (b) + 엔트로피 정규화, (c) + 사용률 균등화, (d) + EMA 패널티/특화, (e) + 직교화. 각 단계별 `avg_routing_entropy`, `avg_expert_cv`, 히트맵 개선을 비교.
- 기대 효과: (b)에서 라우팅 엔트로피 상승, (c)에서 장기적 사용률 편차 완화, (d)에서 단기 쏠림 억제 및 특화 가속, (e)에서 기능 중복 감소.
- 성능 검증: 균형화 손실이 언어 과제에서도 성능 저하 없이(또는 개선되며) 수렴을 안정화하는지 평가.

### 8. 한계와 논의
- 손실 계수의 상호작용으로 초기 튜닝 비용이 존재. 과도한 엔트로피/균등화는 라우팅을 무작위화하여 성능 저하 가능.
- EMA 패널티는 휴리스틱 성격이므로 데이터 분포가 급변할 때 반응 속도가 제한될 수 있음(\(\alpha\) 조정으로 완화).
- 대규모 전문가 수에서 직교화 손실의 계산 비용/메모리 고려 필요.

### 9. 결론
G3MoE는 라우팅 균형과 전문가 특화를 동시에 달성하기 위한 실용적 MoE 프레임워크다. Switch 계열 보조 손실을 확장하여 라우팅 엔트로피와 평균 사용률까지 직접적으로 규제하고, EMA 기반 패널티와 특화 추적으로 단기/장기 균형을 보완한다. 모니터링 지표와 튜닝 가이드를 통해 실제 학습에서 발생하는 붕괴/불균형을 조기에 탐지하고 교정할 수 있음을 보였다.

### 참고 문헌
- [GRIN-MoE] GRIN-MoE: Training Sparse Mixture-of-Experts with Gradient-Informed Routing, arXiv:2409.12136, 2024. (SparseMixer-v2, Heun 기반 라우팅 그래디언트 추정, 토큰 드랍 없이 라우팅 최적화 등 제안)
- [SparseMixer-Enc] Sparse Mixer: Combining Sparse MoE and Linear Mixing for Efficient Encoders, arXiv:2205.12399, 2022. (인코더 효율화 모델로 본 논문의 라우팅용 SparseMixer-v2와는 별개)

### 부록 A. 구현 메모
- 손실 구현: `models/g3moe_model.py`의 `load_balancing_loss_func`에 `router_entropy_coef`, `usage_uniformity_coef` 인자를 추가하여 학습 시 자동으로 합산.
- 설정값: `models/g3moe_config.py`에 `router_entropy_coef`, `usage_uniformity_coef`가 추가되어 구성 파일에서 직접 제어 가능.
- 모니터링: `eval/moe_monitoring_callback.py`의 지표를 활용해 라우팅 문제를 실시간으로 확인.

### 부록 B. 하이퍼파라미터 권장값(초기)
- `router_aux_loss_coef`: 1e-3
- `router_z_loss_coef`: 1e-4 ~ 5e-4
- `router_entropy_coef`: 1e-3 ~ 5e-3
- `usage_uniformity_coef`: 1e-4 ~ 1e-3
- `balancing_strength`: 0.01
- `ema_alpha`: 0.99
- `ortho_loss_coef`: 1e-2


