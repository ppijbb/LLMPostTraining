"""
Configuration management for GRPO training using TRL's GRPOConfig
"""

import logging

logger = logging.getLogger(__name__)

# Import GRPOConfig from TRL
from trl import GRPOConfig


def create_grpo_config(
    model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit",
    output_dir: str = "./grpo_outputs",
    **kwargs) -> GRPOConfig:
    """TRL 표준 GRPOConfig 생성"""
    logger.info(f"🔧 Creating TRL standard GRPOConfig for model: {model_name}")

    # TRL 표준 기본 설정만 사용
    config = GRPOConfig(
        **kwargs  # 사용자 정의 설정으로 기본값 오버라이드
    )

    logger.info("✅ TRL standard GRPOConfig created successfully")
    return config


def create_quick_test_config() -> GRPOConfig:
    """빠른 테스트를 위한 설정"""
    return create_grpo_config(
        model_name="unsloth/Qwen3-0.6B-bnb-4bit",
        output_dir="/mls/conan/training_logs/test_grpo_outputs",
        max_steps=10,
        logging_steps=1,
        save_steps=5,
        eval_steps=5,
    )


def create_production_config(model_name: str = "unsloth/Qwen3-0.6B-bnb-4bit") -> GRPOConfig:
    """프로덕션 학습을 위한 설정"""
    return create_grpo_config(
        model_name=model_name,
        output_dir=f"/mls/conan/training_logs/grpo_outputs_{model_name}",
        max_steps=-1,
        logging_steps=50,
        save_steps=1000,
        eval_steps=1000,
    )


