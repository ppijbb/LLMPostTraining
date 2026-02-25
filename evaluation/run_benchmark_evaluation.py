#!/usr/bin/env python3
"""
G3MoE 모델의 벤치마크 평가 스크립트 (deepeval 사용)
"""
import os
import sys
import json
import torch
import argparse

from typing import List, Any
from PIL import Image
import requests
import re
from tqdm import tqdm
import outlines
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList, MaxLengthCriteria
from peft.peft_model import PeftModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.hellaswag.hellaswag import HellaSwag
from datasets import load_dataset

# 상위 디렉토리를 경로에 추가하여 사용자 정의 모듈 임포트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable torch.compile COMPLETELY to avoid data-dependent branching issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch.compiler.disable()

# Additional safety: reset any existing dynamo state
torch._dynamo.reset()

from models import G3MoEForCausalLM, G3MoEConfig


class G3MoEModelForDeepEval(DeepEvalBaseLLM):
    def __init__(
        self, 
        model_path: str, 
        training_config_path: str | None = None
    ):
        # LoRA 모델의 경우 training_config_path가 필수임을 확인
        if os.path.exists(os.path.join(model_path, 'adapter_config.json')) and not training_config_path:
            raise ValueError("LoRA 모델을 평가하려면 --training_config_path 인자가 반드시 필요합니다.")
        
        self.model_path = model_path
        self.training_config_path = training_config_path
        self.model = None
        self.tokenizer = None
        self.actual_tokenizer = None
        super().__init__()

    def load_model(self):
        if self.model is not None:
            return self.model

        # Force disable torch.compile before model loading
        torch._dynamo.reset()
        torch.compiler.disable()
        os.environ["TORCH_COMPILE_DISABLE"] = "1" 
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        
        print(f"모델 로딩 시작: {self.model_path}")
        
        try:
            tokenizer = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            print("✅ AutoProcessor로 토크나이저 로드 성공")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            print("✅ AutoTokenizer로 토크나이저 로드 성공")
        
        self.tokenizer = tokenizer
        self.actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        if self.actual_tokenizer.pad_token_id is None:
            self.actual_tokenizer.pad_token_id = self.actual_tokenizer.eos_token_id
            print("pad_token_id를 eos_token_id로 설정")

        adapter_config_path = os.path.join(self.model_path, 'adapter_config.json')
        is_lora = os.path.exists(adapter_config_path)
        
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        if is_lora:
            print("LoRA 어댑터 감지. 훈련 설정을 사용하여 베이스 모델을 구성합니다.")
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config["base_model_name_or_path"]
            print(f"  - 베이스 모델 경로: {base_model_path}")
            print(f"  - 훈련 설정 파일: {self.training_config_path}")

            with open(self.training_config_path, 'r') as f:
                train_config = json.load(f)
            g3moe_params = train_config["model_config"]["g3moe_params"]

            base_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
            
            # [MODIFIED]
            # v--
            # 베이스 모델의 설정을 복사하여 사용
            final_config_dict = base_config.to_dict()

            # g3moe 훈련 파라미터로 덮어쓰기
            g3moe_update_params = {
                "n_shared_experts": g3moe_params.get("n_shared_experts"),
                "n_routed_experts": g3moe_params.get("n_routed_experts"),
                "n_group": g3moe_params.get("n_group"),
                "topk_group": g3moe_params.get("topk_group"),
                "num_experts_per_tok": g3moe_params.get("num_experts_per_tok"),
                "first_k_dense_replace": g3moe_params.get("first_k_dense_replace", 8),
                "router_aux_loss_coef": g3moe_params.get("router_aux_loss_coef", 0.001),
                "router_jitter_noise": g3moe_params.get("router_jitter_noise", 0.01),
                "input_jitter_noise": g3moe_params.get("input_jitter_noise", 0.01),
                "rope_scaling": { "rope_type": "linear", "factor": g3moe_params.get("rope_scaling_factor", 1.0) },
                "use_bfloat16": True, # 평가 시에는 bfloat16 사용을 권장
            }
            
            # G3MoEConfig는 text_config, vision_config 등을 인자로 받음
            # 베이스 모델 설정에 text_config가 이미 있는지 확인
            if "text_config" in final_config_dict:
                # Gemma, G3MoE 같은 모델은 text_config 내에 설정이 있음
                final_config_dict["text_config"].update(g3moe_update_params)
            else:
                # LLaMA 같은 모델은 최상위 레벨에 설정이 있음
                # text_config를 새로 만들어줌
                text_config_dict = final_config_dict.copy()
                text_config_dict.update(g3moe_update_params)
                final_config_dict["text_config"] = text_config_dict

            # G3MoEConfig가 멀티모달 관련 파라미터를 요구할 수 있으므로, 없으면 기본값 설정
            final_config_dict.setdefault("vision_config", None)
            final_config_dict.setdefault("boi_token_index", None)
            final_config_dict.setdefault("eoi_token_index", None)
            final_config_dict.setdefault("image_token_index", None)
            
            config = G3MoEConfig(**final_config_dict)
            # --^
            # [MODIFIED]
            
            # Disable torch.compile at model creation time
            torch._dynamo.reset()
            torch.compiler.disable()
            base_model = G3MoEForCausalLM.from_pretrained(
                base_model_path,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map=device_map,
                # attn_implementation="flash_attention_2",
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            print("✅ 훈련 설정을 반영한 LoRA 모델 로드 완료")
        else:
            print("전체 모델 로딩을 시도합니다.")
            self.model = G3MoEForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map=device_map,
                # attn_implementation="flash_attention_2",
            )
            print("✅ 전체 모델 로드 완료")

        self.model.eval()
        return self.model

    @torch.inference_mode()
    def generate(self, prompt: str, *args, **kwargs) -> str:
        if self.model is None:
            self.load_model()
        
        messages = [{"role": "user", "content": prompt}]
        
        if "schema" in kwargs:
            prompt = self.actual_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_dict=False
            )
            schema = kwargs["schema"]
            generator = outlines.models.transformers(self.model, schema=schema)
            response = generator.generate.json(prompt)
            print(response)
        else:
            input_ids = self.actual_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)
            # print(input_ids)
            # del input_ids["pixel_values"]
            # Get the input length for later slicing
            input_length = input_ids["input_ids"].shape[1]
            generator = self.model
            outputs = generator.generate(
                **input_ids,
                generation_config=GenerationConfig(
                    eos_token_id=self.actual_tokenizer.eos_token_id,
                    pad_token_id=self.actual_tokenizer.pad_token_id,
                    do_sample=False,
                ),
                stopping_criteria=StoppingCriteriaList([
                    StopStringCriteria(
                        tokenizer=self.actual_tokenizer,
                        stop_strings=["<end_of_turn>"]
                    ),
                    MaxLengthCriteria(
                        max_length=128,
                        max_position_embeddings=128
                    )
                ])
            )
            
            response_ids = outputs[0][input_length:]
            response = self.actual_tokenizer.decode(response_ids, skip_special_tokens=False).strip()
            print("Response: ", response)
        return response

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_path

@torch.inference_mode()
def run_mme_evaluation(model, tokenizer):
    """
    MME 벤치마크를 수동으로 실행하여 모델의 비전-언어 능력을 평가합니다.
    """
    print("\n" + "="*50)
    print("MME (Multimodal Model Evaluation) 수동 평가 시작")
    print("="*50)

    try:
        print("MME 데이터셋 로딩 중... (최초 실행 시 시간이 걸릴 수 있습니다)")
        mme_dataset = load_dataset("MMMU/MME")
        print("MME 데이터셋 로드 완료.")
    except Exception as e:
        print(f"MME 데이터셋 로딩 실패: {e}. 'pip install datasets'를 시도해보세요.")
        return

    tasks_to_run = ['color', 'count', 'position', 'posters', 'ocr']
    results = {}
    device = model.device

    for task in tasks_to_run:
        if task not in mme_dataset:
            print(f"경고: MME 데이터셋에 '{task}' 태스크가 없습니다. 건너뜁니다.")
            continue
            
        print(f"\n--- '{task}' 태스크 평가 중 ---")
        task_dataset = mme_dataset[task]
        correct_predictions = 0
        total_samples = 0

        for sample in tqdm(task_dataset, desc=f"Evaluating {task}"):
            image = sample['image']
            question = sample['question']
            ground_truth = sample['answer'].strip().lower()

            prompt = f"<image>\n{question}\nAnswer with Yes or No."
            
            inputs = tokenizer(text=[prompt], images=[image], return_tensors="pt").to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            response_text = tokenizer.decode(generated_ids[0])
            cleaned_prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            answer_text = response_text.replace(cleaned_prompt, '').strip().lower()

            match = re.search(r'\b(yes|no)\b', answer_text)
            predicted_answer = match.group(0) if match else "n/a"
            
            if predicted_answer == ground_truth:
                correct_predictions += 1
            total_samples += 1
        
        if total_samples > 0:
            accuracy = (correct_predictions / total_samples) * 100
            results[task] = accuracy
            print(f"'{task}' 태스크 정확도: {accuracy:.2f}% ({correct_predictions}/{total_samples})")

    print("\n" + "="*50)
    print("MME 평가 요약")
    print("="*50)
    if results:
        for task, accuracy in results.items():
            print(f"  - {task:<10}: {accuracy:.2f}%")
        overall_accuracy = sum(results.values()) / len(results)
        print(f"\n  - {'평균':<10}: {overall_accuracy:.2f}%")
    else:
        print("실행된 MME 평가가 없습니다.")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="G3MoE 모델 벤치마크 평가 스크립트")
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
        help="LoRA 모델 평가 시 참조할 훈련 설정 JSON 파일 경로. (예: sft/config/g3moe_deepspeed_config.json)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs='+',
        default=['mmlu', 'hellaswag', 'mme'],
        help="실행할 벤치마크 목록 (mmlu, hellaswag, mme)"
    )
    args = parser.parse_args()

    try:
        eval_model = G3MoEModelForDeepEval(args.model_path, args.training_config_path)
    except ValueError as e:
        print(f"❌ 설정 오류: {e}")
        return
        
    print("\n" + "="*50)
    print("벤치마크 평가 시작")
    print(f"  - 모델: {args.model_path}")
    print(f"  - 벤치마크: {args.benchmarks}")
    print("="*50 + "\n")

    # DeepEval을 사용한 텍스트 기반 벤치마크 실행
    deepeval_benchmarks = []
    if 'mmlu' in args.benchmarks:
        print(" MMLU 벤치마크 추가...")
        deepeval_benchmarks.append(MMLU(n_shots=3))
    if 'hellaswag' in args.benchmarks:
        print(" HellaSwag 벤치마크 추가...")
        deepeval_benchmarks.append(HellaSwag(n_shots=3))

    if deepeval_benchmarks:
        for benchmark in deepeval_benchmarks:
            print(f"--- {benchmark.__class__.__name__} 평가 시작 ---")
            try:
                # DeepEval의 evaluate 메소드는 동기적으로 실행됩니다.
                benchmark.evaluate(model=eval_model)
                print(f"✅ {benchmark.__class__.__name__} 평가 완료")
                print(f"  - 전체 점수: {benchmark.overall_score:.4f}")

            except Exception as e:
                print(f"❌ {benchmark.__class__.__name__} 평가 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            print("-" * (len(benchmark.__class__.__name__) + 15) + "\n")
    
    # MME (비전) 벤치마크 수동 실행
    if 'mme' in args.benchmarks:
        model = eval_model.load_model()
        tokenizer = eval_model.tokenizer
        run_mme_evaluation(model, tokenizer)

    print("--- 모든 벤치마크 평가 완료 ---")

if __name__ == "__main__":
    main() 