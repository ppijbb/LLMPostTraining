# GRPO (Group Relative Policy Optimization) Training

TRL의 표준 GRPO 트레이너를 사용하여 효율적인 강화 학습을 수행하는 모듈입니다. TRL의 표준 데이터 형식과 설정을 따릅니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# TRL 표준 의존성 설치
pip install torch transformers datasets accelerate
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl wandb

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

### 2. 기본 훈련

```bash
# 빠른 테스트
python train_grpo.py --quick-test

# 단일 통합 보상 함수로 훈련 (기본)
python train_grpo.py --reward-function single

# 다중 컴포넌트 보상 함수로 훈련
python train_grpo.py --reward-function multi

# 기본 훈련 (TRL 표준 데이터셋 사용)
python train_grpo.py --max-samples 1000

# 프로덕션 훈련 (생성 로깅 포함)
python train_grpo.py --production
```

### 3. 커스텀 데이터로 훈련

```bash
# JSONL 파일 사용 (TRL 표준 형식)
python train_grpo.py --custom-data /path/to/your_data.jsonl

# 데이터 형식 예시:
# {"prompt": "질문", "chosen": "선호 답변", "rejected": "비선호 답변"}
```

# 단일 통합 보상 함수 사용 (기본)
python train_grpo.py --reward-function single

# 다중 컴포넌트 보상 함수 사용
python train_grpo.py --reward-function multi

# 설정 파일과 함께 사용
python train_grpo.py --reward-function single --reward-config config/reward_config.json
```


## 📊 지원하는 모델

- `llama-3.1-8b`: Llama 3.1 8B (기본)
- `llama-3.1-70b`: Llama 3.1 70B
- `gemma-2-9b`: Gemma 2 9B
- `qwen2.5-7b`: Qwen2.5 7B

## 📦 지원하는 데이터셋

- `ultrafeedback`: HuggingFaceH4/ultrafeedback_binarized (기본)
- `ultrafeedback_large`: 더 큰 UltraFeedback 데이터셋
- `hh_rlhf`: Anthropic/hh-rlhf
- `openai_summarize`: OpenAI summarize from feedback

## ⚙️ 설정 옵션

### 명령줄 옵션

```bash
python train_grpo.py [OPTIONS]

옵션:
  --model {llama-3.1-8b,llama-3.1-70b,gemma-2-9b,qwen2.5-7b}
                        사용할 모델 (기본: llama-3.1-8b)
  --dataset {ultrafeedback,ultrafeedback_large,hh_rlhf,openai_summarize}
                        사용할 데이터셋 (기본: ultrafeedback)
  --custom-data PATH    커스텀 데이터셋 파일 경로
  --config PATH         설정 파일 경로
  --quick-test          빠른 테스트 실행
  --production          프로덕션 훈련 실행
  --max-samples N       사용할 최대 샘플 수
  --epochs N            훈련 에포크 수
  --learning-rate LR    학습률
  --batch-size N        배치 크기
  --output-dir PATH     출력 디렉토리
  --reward-function {single,multi}
                        보상 함수 타입 (기본: single)
  --reward-config PATH  보상 함수 설정 파일 경로 (JSON)
                        커스텀 보상 함수 설정 파일
  --wandb-project NAME  Weights & Biases 프로젝트 이름
  --no-wandb           Weights & Biases 비활성화
  --eval-only          평가만 실행
  --resume-from-checkpoint PATH
                        체크포인트에서 재시작
  --enable-generation-logging
                        생성 로깅 활성화 (기본값: 활성화)
  --disable-generation-logging
                        생성 로깅 비활성화
  --generation-log-dir PATH
                        생성 로그 저장 디렉토리 (기본값: {output_dir}/generation_logs)
  --max-generation-samples N
                        로깅을 위한 최대 생성 샘플 수 (기본값: 5)
```

### Generation Logging

학습 중 evaluation 단계에서 생성된 텍스트를 로그로 확인할 수 있습니다:

```bash
# 기본적으로 활성화됨
python train_grpo.py --reward-function single

# 비활성화
python train_grpo.py --disable-generation-logging

# 커스텀 로그 디렉토리 설정
python train_grpo.py --generation-log-dir ./my_logs

# 생성 샘플 수 조절
python train_grpo.py --max-generation-samples 10
```

로그 파일은 `{output_dir}/generation_logs/` 디렉토리에 `generation_log_step_{n}.json` 형태로 저장됩니다.

로그 파일 예시:
```json
[
  {
    "eval_step": 1,
    "sample_index": 0,
    "prompt": "What is the capital of France?",
    "generated": "The capital of France is Paris. It is located in...",
    "full_response": "The capital of France is Paris. It is located in the north-central part of the country..."
  },
  {
    "eval_step": 1,
    "sample_index": 1,
    "prompt": "Explain quantum computing in simple terms.",
    "generated": "Quantum computing uses quantum mechanics to perform...",
    "full_response": "Quantum computing uses quantum mechanics to perform calculations..."
  }
]
```

### 설정 파일 예제

```json
{
  "model_name": "unsloth/llama-3.1-8b-bnb-4bit",
  "max_seq_length": 2048,
  "learning_rate": 5e-7,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 2,
  "per_device_eval_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_samples": 1000,
  "output_dir": "./grpo_outputs",
  "beta": 0.1,
  "gamma": 1.0,
  "group_size": 4,
  "reward_function_type": "systematic",
  "reward_config_name": "balanced"
}
```

### 보상 함수 설정 예제

`config/reward_config.json`:
```json
{
  "accuracy_weight": 0.4,
  "length_weight": 0.2,
  "quality_weight": 0.4,
  "accuracy": {
    "correct_keywords": ["correct", "right", "yes", "정확", "맞아"]
  },
  "length": {
    "optimal_length": 150,
    "length_weight": 0.1
  },
  "quality": {
    "reward_scale": 1.0,
    "penalty_scale": -0.5,
    "quality_keywords": ["좋아", "완벽", "우수"],
    "negative_keywords": ["모르겠", "잘못", "틀렸"]
  }
}
```

## 📁 데이터 형식

### UltraFeedback 형식

```json
{
  "chosen": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ],
  "rejected": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "I don't know."}
  ]
}
```

### 커스텀 데이터 형식

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I don't know."
}
```

## 🔧 고급 사용법

### 1. 커스텀 설정 생성

```python
from config import create_default_config

# 기본 설정 생성
config = create_default_config("llama-3.1-8b", "ultrafeedback")

# 커스텀 설정 적용
config.learning_rate = 1e-6
config.num_train_epochs = 5
config.max_samples = 5000

# 설정 저장
from config import save_config_to_file
save_config_to_file(config, "my_config.json")
```

### 2. 프로그래밍 방식으로 훈련

```python
from grpo_trainer import GRPOTrainer, GRPOConfig
from data_loader import create_grpo_dataloader

# 설정 생성
config = GRPOConfig(
    model_name="unsloth/llama-3.1-8b-bnb-4bit",
    max_samples=1000,
    output_dir="./my_grpo_outputs"
)

# 데이터 로더 생성
data_loader, dataset = create_grpo_dataloader(
    model_name=config.model_name,
    dataset_name="HuggingFaceH4/ultrafeedback_binarized",
    max_samples=config.max_samples
)

# 훈련기 생성 및 훈련
trainer = GRPOTrainer(config)
trainer.load_model()
trainer.train(dataset)
trainer.save_model()
```

## 📊 모니터링

### Weights & Biases

```bash
# W&B 프로젝트 설정
python train_grpo.py --wandb-project my-grpo-project

# W&B 비활성화
python train_grpo.py --no-wandb
```

### 로그 파일

훈련 로그는 `grpo_training.log` 파일에 저장됩니다.

## 🐛 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   python train_grpo.py --batch-size 1
   
   # 그래디언트 누적 사용
   # config.py에서 gradient_accumulation_steps 증가
   ```

2. **모델 로딩 실패**
   ```bash
   # 4bit 양자화 비활성화
   # config.py에서 load_in_4bit=False로 설정
   ```

3. **데이터셋 로딩 실패**
   ```bash
   # 스트리밍 모드 사용
   # data_loader.py에서 streaming=True로 설정
   ```

### 로그 확인

```bash
# 실시간 로그 확인
tail -f grpo_training.log

# 에러 로그만 확인
grep "ERROR" grpo_training.log
```

## 📚 참고 자료

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [GRPO Paper](https://arxiv.org/abs/2406.05896)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [TRL Library](https://huggingface.co/docs/trl)

## 🤝 기여하기

1. 이슈 리포트
2. 포크 및 브랜치 생성
3. 변경사항 커밋
4. 풀 리퀘스트 생성

## 📄 라이선스

MIT License
