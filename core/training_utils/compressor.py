import os
from pathlib import Path
from typing import Optional

import torch  # type: ignore
from dotenv import load_dotenv
from huggingface_hub import HfApi  # type: ignore
from llmcompressor import oneshot  # type: ignore
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)

# .env 파일 로드
load_dotenv()

# Gemma3 공식 예제: https://github.com/vllm-project/llm-compressor/blob/main/examples/multimodal_vision/gemma3_example.py
# AWQ는 Gemma3에서 알려진 이슈(linear weight / NaN) 있음 → GPTQ 사용 권장 (issue #2102)
DATASET_ID = "flickr30k"
NUM_CALIBRATION_SAMPLES = 768
MAX_SEQUENCE_LENGTH = 2048
DATASET_SPLIT = "test"


def calc_state_dict_params(module: torch.nn.Module) -> int:
    """
    주어진 모듈의 state_dict 항목들을 순회하여 파라미터(요소) 수를 합산합니다.
    Tensor 항목은 .numel()로 계산하고, 다른 객체에 대해선 .numel()가 있으면 사용합니다.
    안전하게 예외를 처리하여 실패 시 해당 항목은 건너뜁니다.
    """
    total = 0
    try:
        sd = module.state_dict()
    except Exception:
        # state_dict 접근 불가 시 0 반환
        return 0
    for v in sd.values():
        try:
            if isinstance(v, torch.Tensor):
                total += int(v.numel())
            else:
                # 다른 객체가 numel을 제공하면 사용
                if hasattr(v, "numel"):
                    total += int(v.numel())
                # 그 외는 무시
        except Exception:
            # 안전하게 진행
            pass
    return total


def format_params(n: int) -> str:
    """
    파라미터 수를 사람이 읽기 쉬운 문자열로 변환.
    예: "1,234,567 params (1.23M)"
    """

    def abbrev(num: int) -> str:
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        if num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        if num >= 1_000:
            return f"{num / 1_000:.2f}K"
        return str(num)

    return f"{n:,} params ({abbrev(n)})"

# /home/conan/workspace/llm_training/sft/config/chat_template.txt"
def _resolve_chat_template_path() -> Optional[Path]:
    """프로젝트 내 chat_template 파일 경로 반환 (있을 경우)."""
    candidates: list[Path] = [
        Path(__file__).resolve().parent.parent
        / "sft"
        / "config"
        / "chat_template.txt",
    ]
    env_path = os.getenv("CHAT_TEMPLATE_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    for p in candidates:
        if p.is_file():
            return p
    return None


def quantize_gemma3(
    model_name: str,
    save_directory: str,
    bits: int = 4,
    group_size: int = 128,
):
    """
    Gemma3를 GPTQ(W4A16)로 양자화합니다.
    AWQ는 Gemma3에서 'linear has no attribute weight' / NaN 이슈가 있어 공식 예제대로 GPTQ 사용.
    """
    # 1. 원본 모델·프로세서 로드
    print(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    template_path = _resolve_chat_template_path()
    if template_path is not None:
        processor.chat_template = template_path.read_text()

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=False,
    )

    orig_params = 0
    try:
        orig_params = calc_state_dict_params(model)
        print("Model parameters (before quantization):", format_params(orig_params))
    except Exception as e:
        print(f"Could not compute original model parameter count: {e}")

    # 2. GPTQ 양자화 (Gemma3 공식 예제와 동일 레시피)
    print("Starting GPTQ quantization (Gemma3 recommended)...")
    recipe = [
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=[
                "lm_head",
                r"re:model\.vision_tower.*",
                r"re:model\.multi_modal_projector.*",
            ],
        ),
    ]

    oneshot(
        model=model,
        processor=processor,
        dataset=DATASET_ID,
        splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
        recipe=recipe,
        shuffle_calibration_samples=False,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        trust_remote_code_model=True,
    )

    # 3. 생성용 디스패치 후 저장 (순서 중요: dispatch → 테스트 → save)
    dispatch_for_generation(model)
    model.eval()

    try:
        quant_params = calc_state_dict_params(model)
        print("Model parameters (after quantization):", format_params(quant_params))
        if orig_params:
            print(
                f"Quantized: {format_params(orig_params)} -> {format_params(quant_params)}"
            )
    except Exception as e:
        print(f"Could not compute quantized parameter count: {e}")

    # 4. 저장 전 생성 테스트 (NaN/Inf 방지 확인)
    try:
        with torch.no_grad():
            print("##### [ Test Generation (before save) ] #####")
            inputs = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "안녕하세요. 테스트입니다. 짧게 답해주세요.",
                            }
                        ],
                    }
                ],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                tokenize=True,
                add_special_tokens=False,
                return_dict=True,
                add_generation_prompt=False,
            )
            generated = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_new_tokens=10,
                do_sample=False,
                disable_compile=True,
            )
            out = processor.batch_decode(generated.cpu(), skip_special_tokens=True)
            if isinstance(out, list):
                out = out[0]
            if not out or "nan" in out.lower() or "inf" in out.lower():
                raise ValueError("Generation produced NaN/Inf or empty output.")
            print(out)
    except Exception as e:
        print(f"Generation test failed: {e}")
        return None

    model.save_pretrained(save_directory, save_compressed=True)
    processor.save_pretrained(save_directory)
    return model


@torch.inference_mode
def model_test(save_directory: str) -> None:
    """
    저장된 GPTQ(compressed-tensors) 모델을 로드해 생성 테스트.
    config 기반으로 클래스를 잡으려 AutoModelForCausalLM 사용.
    """
    save_path = Path(save_directory)
    if not save_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {save_directory}. "
            "Run quantize_gemma3 first or set QUANTIZE_SAVE_DIR / pass --test-only <path>."
        )

    print("##### [ Test Generation (load from disk) ] #####")
    # 양자화 체크포인트는 config에 quantization_config 있으면 자동으로 decompress 로드
    model = AutoModelForCausalLM.from_pretrained(
        save_directory,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    dispatch_for_generation(model)
    model.eval()

    processor = AutoProcessor.from_pretrained(save_directory, trust_remote_code=True)
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "안녕하세요. 테스트입니다. 짧게 답해주세요."}
                ],
            }
        ],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )
    generated = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_new_tokens=10,
        do_sample=False,
        disable_compile=True,
    )
    out = processor.batch_decode(generated.cpu(), skip_special_tokens=True)
    if isinstance(out, list):
        out = out[0]    
    print(generated.shape)
    print(out)


def upload(
    save_directory: str,
    hf_repo_name: str,
    token: Optional[str] = None,
):  # 4. Hugging Face Hub에 업로드
    print(f"Uploading to Hugging Face Hub: {hf_repo_name}")

    # Hugging Face 토큰 설정
    if token is None:
        token = os.getenv("HUGGINGFACE_TOKEN")

    if token is None:
        raise ValueError(
            "Hugging Face token is required. Provide it as an argument or set HUGGINGFACE_TOKEN environment variable."
        )

    # 레포지토리 생성 및 업로드
    api = HfApi(token=token)

    # 레포지토리가 존재하지 않으면 생성
    try:
        api.create_repo(repo_id=hf_repo_name, private=False, exist_ok=True)
    except Exception as e:
        print(f"Repository creation failed (may already exist): {e}")

    # 파일 업로드
    api.upload_folder(
        folder_path=save_directory, repo_id=hf_repo_name, repo_type="model", token=token
    )

    print(f"Successfully uploaded quantized model to {hf_repo_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemma3 GPTQ quantization and test")
    parser.add_argument(
        "--test-only",
        type=str,
        nargs="?",
        const="",
        metavar="PATH",
        help="Only run model_test on saved model. Path = dir or env QUANTIZE_SAVE_DIR.",
    )
    args = parser.parse_args()

    SAVE_DIR = os.getenv("QUANTIZE_SAVE_DIR", "/mls/conan/quantized_model")
    test_dir = (args.test_only if args.test_only else None) or SAVE_DIR

    if args.test_only is not None:
        model_test(save_directory=test_dir)
        raise SystemExit(0)

    MODEL_NAME = os.getenv("QUANTIZE_MODEL_NAME", "Gunulhona/Gemma-3-4B")
    HF_REPO_NAME = os.getenv("QUANTIZE_HF_REPO", "Gunulhona/Gemma-3-4B-w4a16")
    print("=============================================")
    print(f"Quantizing model: {MODEL_NAME} to {SAVE_DIR}")
    print("=============================================")
    model = quantize_gemma3(
        model_name=MODEL_NAME,
        save_directory=SAVE_DIR,
        bits=4,
        group_size=128,
    )

    if model is not None:
        model_test(save_directory=SAVE_DIR)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            upload(save_directory=SAVE_DIR, hf_repo_name=HF_REPO_NAME, token=token)
        else:
            print("HF_TOKEN not set; skipping upload.")

    # model_test(save_directory=SAVE_DIR)
    # token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    # if token:
    #     upload(save_directory=SAVE_DIR, hf_repo_name=HF_REPO_NAME, token=token)