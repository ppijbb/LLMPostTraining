# G3MoE SFT Training

이 디렉토리는 TRL (Transformers Reinforcement Learning)을 사용하여 G3MoE 모델을 Supervised Fine-Tuning(SFT)하는 스크립트들을 포함합니다.

## 파일 구조

```
sft/
├── custom_model_sft.py          # 메인 SFT 훈련 스크립트
├── run_g3moe_sft.sh            # 훈련 실행 셸 스크립트
├── config/
│   └── g3moe_sft_config.json   # 훈련 설정 파일
└── README.md                   # 이 파일
```

## 필요한 패키지

```bash
pip install torch transformers trl peft datasets wandb accelerate
```

또는 프로젝트 루트에서:

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 기본 실행

**일반 훈련 (기본 컨텍스트):**
```bash
# 기본 설정으로 실행
./sft/run_g3moe_sft.sh

# 커스텀 설정 파일로 실행
./sft/run_g3moe_sft.sh path/to/your/config.json

# 훈련 후 평가까지 수행
./sft/run_g3moe_sft.sh sft/config/g3moe_sft_config.json --eval
```

**🚀 DeepSpeed + 120K 컨텍스트 훈련 (80GB GPU 최적화):**
```bash
# 단일 GPU 120K 컨텍스트 훈련
./sft/run_g3moe_deepspeed.sh sft/config/g3moe_120k_deepspeed_config.json

# 멀티 GPU 120K 컨텍스트 훈련 (2 GPUs)
./sft/run_g3moe_deepspeed.sh sft/config/g3moe_120k_deepspeed_config.json 2

# ZeRO-2 사용 (일반 데이터셋)
./sft/run_g3moe_deepspeed.sh sft/config/g3moe_large_dataset_config.json

# ZeRO-3 사용 (최대 메모리 절약)
./sft/run_g3moe_deepspeed.sh sft/config/g3moe_120k_deepspeed_config.json
```

### 2. Python 스크립트 직접 실행

```bash
# 기본 설정으로 실행
python sft/custom_model_sft.py

# JSON 설정 파일 사용
python sft/custom_model_sft.py sft/config/g3moe_sft_config.json

# 명령행 인자 사용
python sft/custom_model_sft.py \
    --model_name_or_path google/gemma-3-4b-it \
    --output_dir ./outputs/g3moe_sft \
    --dataset_name Gunulhona/open_m_3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --use_lora True \
    --lora_r 32 \
    --bf16 True
```

## Google 공식 Gemini Fine-tuning 가이드라인 적용

Google Cloud의 [Gemini fine-tuning 가이드](https://medium.com/google-cloud/fine-tuning-gemini-best-practices-for-data-hyperparameters-and-evaluation-65f7c7b6b15f)를 기반으로 한 권장 하이퍼파라미터:

### 📊 데이터셋 크기별 권장 설정

**소규모 데이터셋 (< 1000개 예제, 평균 컨텍스트 < 500토큰):**
```bash
./sft/run_g3moe_sft.sh sft/config/g3moe_small_dataset_config.json
```
- `epochs = 20`
- `learning_rate = 2e-4` (Google 권장 multiplier 10x 적용)
- `lora_r = 4` (adapter size)
- `max_seq_length = 1024`

**대규모 데이터셋 (>= 1000개 예제 또는 평균 컨텍스트 >= 500토큰):**
```bash
./sft/run_g3moe_sft.sh sft/config/g3moe_large_dataset_config.json
```
- `epochs = 10`
- `learning_rate = 1e-4` (기본 또는 5x multiplier)
- `lora_r = 8` (더 큰 adapter size)
- `max_seq_length = 2048`

### 🎯 핵심 원칙

1. **품질 > 양**: Google은 고품질의 작은 데이터셋이 큰 노이즈 데이터셋보다 효과적이라고 강조
2. **LoRA 사용**: Vertex AI는 LoRA를 활용하여 적은 파라미터로 효과적인 fine-tuning 수행
3. **복잡한 예제 중심**: 기본 모델이 어려워하는 예제에 집중
4. **검증 데이터셋 필수**: 과적합 방지를 위한 별도 검증 세트 유지

## 설정 옵션

### 모델 설정 (ModelArguments)

- `model_name_or_path`: 사전 훈련된 모델 경로 또는 HF Hub ID
- `tokenizer_name_or_path`: 토크나이저 경로 (None이면 모델 경로 사용)
- `use_lora`: LoRA 사용 여부 (기본값: True)
- `lora_r`: LoRA rank (기본값: 16)
- `lora_alpha`: LoRA alpha 파라미터 (기본값: 32)
- `lora_dropout`: LoRA dropout (기본값: 0.1)
- `trust_remote_code`: 원격 코드 신뢰 여부 (기본값: True)

### 데이터 설정 (DataArguments)

- `dataset_name`: 사용할 데이터셋 이름 (기본값: "Gunulhona/open_m_3")
- `max_seq_length`: 최대 시퀀스 길이 (기본값: 2048)
- `test_size`: 테스트 데이터 비율 (기본값: 0.1)
- `text_only`: 텍스트만 사용할지 여부 (기본값: False)
- `streaming`: 스트리밍 모드 사용 여부 (기본값: False)

### 훈련 설정 (SFTConfig)

주요 하이퍼파라미터:
- `num_train_epochs`: 훈련 에포크 수 (기본값: 3)
- `per_device_train_batch_size`: 디바이스당 배치 크기 (기본값: 1)
- `gradient_accumulation_steps`: 그래디언트 누적 스텝 수 (기본값: 16)
- `learning_rate`: 학습률 (기본값: 2e-5)
- `weight_decay`: 가중치 감쇠 (기본값: 0.01)
- `lr_scheduler_type`: 학습률 스케줄러 (기본값: "cosine")
- `warmup_ratio`: 웜업 비율 (기본값: 0.03)

## 🔥 80GB GPU + 120K 컨텍스트 최적화

### DeepSpeed ZeRO 사용 (필수)

**ZeRO-2 설정 (기본 권장):**
```json
{
    "deepspeed": "sft/config/deepspeed_zero2.json",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32
}
```

**ZeRO-3 설정 (최대 메모리 절약):**
```json
{
    "deepspeed": "sft/config/deepspeed_zero3.json",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 64
}
```

### 120K 컨텍스트 특화 설정

```json
{
    "max_seq_length": 120000,
    "gradient_checkpointing": true,
    "dataloader_num_workers": 0,
    "dataloader_pin_memory": false,
    "learning_rate": 5e-5,
    "num_train_epochs": 5
}
```

### 메모리 최적화 팁

1. **LoRA 사용**: 메모리 사용량 대폭 감소
2. **CPU Offload**: DeepSpeed ZeRO-3의 optimizer/parameter offload
3. **환경 변수**: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024`
4. **시스템 요구사항**: 최소 64GB RAM 권장

## 모니터링

### Weights & Biases

WandB 로그인 후 설정에서 활성화:

```bash
wandb login
```

```json
{
    "report_to": ["wandb"],
    "run_name": "g3moe-sft-experiment"
}
```

### 로컬 로그

```json
{
    "logging_dir": "./outputs/g3moe_sft/logs",
    "logging_steps": 5,
    "eval_steps": 100,
    "save_steps": 500
}
```

## 멀티GPU 훈련

### DeepSpeed 멀티GPU 훈련

**자동 GPU 감지로 훈련:**
```bash
# 사용 가능한 모든 GPU 사용
./sft/run_g3moe_deepspeed.sh sft/config/g3moe_120k_deepspeed_config.json

# 특정 GPU 개수 지정
./sft/run_g3moe_deepspeed.sh sft/config/g3moe_120k_deepspeed_config.json 4
```

**수동 torchrun 실행:**
```bash
# 4 GPU DeepSpeed 훈련
torchrun --nproc_per_node=4 sft/custom_model_sft.py sft/config/g3moe_120k_deepspeed_config.json

# 특정 GPU만 사용
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 sft/custom_model_sft.py sft/config/g3moe_120k_deepspeed_config.json
```

### DeepSpeed 설정 파일들

- `sft/config/deepspeed_zero2.json`: ZeRO-2 (균형잡힌 성능)
- `sft/config/deepspeed_zero3.json`: ZeRO-3 (최대 메모리 절약 + CPU offload)

## 훈련 성능 모니터링 (Google 가이드라인)

### 📈 핵심 메트릭

**Training Loss & Validation Loss:**
- Training loss는 감소해야 함
- Validation loss가 training loss보다 현저히 높으면 과적합 신호
- 이상적: 두 loss 모두 지속적으로 감소

**Next Token Prediction Accuracy:**
- 시퀀스 예측 정확도가 시간에 따라 증가해야 함

### 🚨 문제 유형별 진단

**1. 성능 부족 (Suboptimal Performance)**
- **증상**: Training/validation loss가 감소하지만 수렴하지 않음
- **원인**: 데이터셋이 너무 작거나 다양성 부족
- **해결책**: 
  - epochs 수 증가 또는 learning rate multiplier 증가
  - 더 많은 고품질 데이터 수집

**2. 과적합 (Overfitting)**
- **증상**: Training loss는 감소하지만 validation loss가 증가
- **원인**: 모델 용량이 데이터 대비 과도함
- **해결책**:
  - epochs 수를 validation loss 최소점까지 감소
  - 훈련 데이터 크기와 다양성 증가

**3. 데이터 문제 (Data Issues)**
- **증상**: 초기 loss가 매우 높음 (>10)
- **원인**: 입력 길이가 최대 컨텍스트 길이 초과로 인한 truncation
- **해결책**: 데이터셋 형식과 길이 재검토

## 트러블슈팅

### 1. CUDA Out of Memory

- 배치 크기 줄이기: `per_device_train_batch_size`를 1로 설정
- 그래디언트 누적 증가: `gradient_accumulation_steps` 증가
- 그래디언트 체크포인팅 활성화
- 시퀀스 길이 줄이기: `max_seq_length` 감소

### 2. 모델 로딩 실패

G3MoE 모델이 아직 공개되지 않은 경우, 스크립트는 자동으로 기본 Gemma 모델로 폴백합니다.

### 3. 데이터셋 로딩 오류

- 인터넷 연결 확인
- HuggingFace Hub 접근 권한 확인
- 로컬 데이터셋 사용 시 경로 확인

## 예제 실행 결과

```
========================================
      G3MoE SFT Training Script        
========================================
Project Root: /path/to/llm_training
Config File: /path/to/config.json
Output Directory: ./outputs/g3moe_sft
CUDA Devices: 0
All dependencies found!
GPU detected: 1 device(s) available
Starting G3MoE SFT training...

Setting up model and tokenizer...
Loaded G3MoE config
Loaded G3MoE model successfully

Setting up dataset...
Loading dataset: Gunulhona/open_m_3
Dataset loaded:
  train: 8000 examples
  test: 2000 examples

Setting up trainer...
Starting training...
Training completed successfully!
Model saved to: ./outputs/g3moe_sft
```

## 추가 리소스

- [TRL 문서](https://huggingface.co/docs/trl/)
- [PEFT 문서](https://huggingface.co/docs/peft/)
- [G3MoE 논문](https://arxiv.org/abs/2501.xxxxx) (예정) 