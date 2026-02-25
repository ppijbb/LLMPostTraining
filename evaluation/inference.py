#!/usr/bin/env python3
"""
G3MoE SFT 모델 추론 스크립트
"""

import os
import sys
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from peft.peft_model import PeftModel

# 상위 디렉토리를 경로에 추가하여 사용자 정의 모듈 임포트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import G3MoEForCausalLM, G3MoEConfig

def load_model_for_inference(model_path: str, training_config_path: str):
    """
    훈련된 모델과 토크나이저를 추론을 위해 로드합니다.
    LoRA 어댑터의 경우, 훈련 설정 파일을 참조하여 정확한 G3MoE 구조를 재현합니다.
    """
    print(f"모델 로딩 시작: {model_path}")

    # 토크나이저 로드
    try:
        tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("✅ AutoProcessor로 토크나이저 로드 성공")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✅ AutoTokenizer로 토크나이저 로드 성공")
    
    # 생성(generation)을 위해 pad 토큰 설정
    actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    if actual_tokenizer.pad_token_id is None:
        actual_tokenizer.pad_token_id = actual_tokenizer.eos_token_id
        print("pad_token_id를 eos_token_id로 설정")

    # LoRA 모델인지 확인
    adapter_config_path = os.path.join(model_path, 'adapter_config.json')
    is_lora = os.path.exists(adapter_config_path)
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    if is_lora:
        if not training_config_path:
            raise ValueError("LoRA 모델을 로드하려면 --training_config_path 인자가 반드시 필요합니다.")

        print("LoRA 어댑터 감지. 훈련 설정을 사용하여 베이스 모델을 구성합니다.")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config["base_model_name_or_path"]
        print(f"  - 베이스 모델 경로: {base_model_path}")
        print(f"  - 훈련 설정 파일: {training_config_path}")

        # 훈련 설정 파일에서 g3moe_params를 로드하여 G3MoEConfig 생성
        with open(training_config_path, 'r') as f:
            train_config = json.load(f)
        g3moe_params = train_config["model_config"]["g3moe_params"]

        base_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        base_model_config = base_config.to_dict()

        g3moe_update_config = {
            "n_shared_experts": g3moe_params["n_shared_experts"],
            "n_routed_experts": g3moe_params["n_routed_experts"],
            "n_group": g3moe_params["n_group"],
            "topk_group": g3moe_params["topk_group"],
            "num_experts_per_tok": g3moe_params["num_experts_per_tok"],
            "first_k_dense_replace": g3moe_params.get("first_k_dense_replace", 8),
            "router_aux_loss_coef": 0.001,
            "router_jitter_noise": 0.01,
            "input_jitter_noise": 0.01,
            "model_type": "g3moe_text",
            "rope_scaling": { "rope_type": "linear", "factor": g3moe_params["rope_scaling_factor"] },
            "use_bfloat16": True,
        }
        
        if "text_config" in base_model_config:
            base_model_config["text_config"].update(g3moe_update_config)
            config = G3MoEConfig(**base_model_config)
        else:
            base_model_config.update(g3moe_update_config)
            config = G3MoEConfig(**base_model_config)

        # 베이스 모델 가중치를 로드하되, 훈련된 G3MoE 구조를 적용합니다.
        base_model = G3MoEForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device_map,
        )
        
        # Peft 모델(어댑터) 로드
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✅ 훈련 설정을 반영한 LoRA 모델 로드 완료")

    else:
        print("전체 모델 로딩을 시도합니다.")
        model = G3MoEForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device_map,
        )
        print("✅ 전체 모델 로드 완료")

    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="G3MoE 모델 추론 스크립트")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="훈련된 모델(어댑터 또는 전체)이 저장된 디렉토리 경로",
    )
    parser.add_argument(
        "--training_config_path",
        type=str,
        default=None,
        help="LoRA 모델 추론 시 참조할 훈련 설정 JSON 파일 경로. (예: sft/config/g3moe_deepspeed_config.json)",
    )
    args = parser.parse_args()

    try:
        model, tokenizer = load_model_for_inference(args.model_path, args.training_config_path)
    except Exception as e:
        print(f"❌ 모델 로딩 중 오류 발생: {e}")
        print("스크립트를 종료합니다.")
        return

    print("\n" + "="*50)
    print("G3MoE 모델 대화형 테스트")
    # ... existing code ... 