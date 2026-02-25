import os
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from transformers import AutoConfig
#from huggingface_hub import configure_http_backend

# GPU 메모리 최적화 설정
torch.cuda.empty_cache()  # GPU 캐시 비우기
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"  # 메모리 할당 크기 증가
torch.backends.cudnn.benchmark = True  # CUDNN 벤치마크 활성화
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 허용 (Ampere GPU 이상)
torch.backends.cudnn.allow_tf32 = True  # CUDNN에서 TF32 허용

# Hugging Face 캐시 경로 설정 (NAS 사용)
os.environ["HF_HOME"] = "/nas/glory/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/nas/glory/hf_cache/models"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# NAS 경로 설정
UNSLOTH_CACHE_DIR = "/nas/glory/hf_cache/unsloth_compiled"
UNSLOTH_CHECKPOINT_DIR = "/nas/glory/hf_cache/unsloth_checkpoints"

# 디렉토리가 없으면 생성
os.makedirs(UNSLOTH_CACHE_DIR, exist_ok=True)
os.makedirs(UNSLOTH_CHECKPOINT_DIR, exist_ok=True)
os.makedirs("/nas/glory/hf_cache", exist_ok=True)
os.makedirs("/nas/glory/hf_cache/models", exist_ok=True)

def load_model_and_tokenizer(model_name="unsloth/gemma-3-4b-it"):
    """
    모델과 토크나이저를 로드하고 설정하는 함수
    """
    # GPU 정보 출력
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 메모리 사용량 (로딩 전): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"총 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # GPU 메모리 계산
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    max_memory_to_use = int(total_gpu_memory * 0.95)  # 총 메모리의 95% 사용
    
    print(f"사용할 최대 메모리: {max_memory_to_use}GB")
    
    # 모델 및 토크나이저 로드
    model, tokenizer = FastModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        load_in_4bit = True,  # 4비트 양자화로 메모리 절약
        load_in_8bit = False,
        full_finetuning = False,
        cache_dir = UNSLOTH_CACHE_DIR,  # NAS의 unsloth 캐시 디렉토리 사용
        local_files_only=False,  # 온라인에서 다운로드 허용
        resume_download=True,  # 중단된 다운로드 이어서 받기
        token=None,  # 토큰이 있다면 여기에 추가
        # device_map="auto",
        max_memory={0: f"{max_memory_to_use}GiB"},  # 계산된 최대 메모리 사용
        #torch_dtype=torch.bfloat16,  # BF16 precision 사용
        attn_implementation="flash_attention_2",  # Flash Attention 2 사용
    )

    print(f"GPU 메모리 사용량 (모델 로드 후): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # PEFT 모델 설정
    # model = FastModel.get_peft_model(
    #     model,
    #     finetune_vision_layers = False,
    #     finetune_language_layers = True,
    #     finetune_attention_modules = True,
    #     finetune_mlp_modules = True,
    #     r = 8,
    #     lora_alpha = 8,
    #     lora_dropout = 0,
    #     bias = "none",
    #     random_state = 3407,
    # )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.05,
        bias = "none",
        random_state = 3407,
    )
    
    print(f"GPU 메모리 사용량 (PEFT 설정 후): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    # 채팅 템플릿 설정
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )

    return model, tokenizer



# def load_model_and_tokenizer(model_name="unsloth/gemma-3-12b-it"):
#     """
#     모델과 토크나이저를 로드하고 설정하는 함수
#     """
#     # 모델 및 토크나이저 로드
#     model, tokenizer = FastModel.from_pretrained(
#         model_name = model_name,
#         max_seq_length = 2048,
#         load_in_4bit = True,
#         load_in_8bit = False,
#         full_finetuning = False,
#     )

#     # PEFT 모델 설정
#     model = FastModel.get_peft_model(
#         model,
#         finetune_vision_layers = False,
#         finetune_language_layers = True,
#         finetune_attention_modules = True,
#         finetune_mlp_modules = True,
#         r = 8,
#         lora_alpha = 8,
#         lora_dropout = 0,
#         bias = "none",
#         random_state = 3407,
#     )

#     # 채팅 템플릿 설정
#     tokenizer = get_chat_template(
#         tokenizer,
#         chat_template = "gemma-3",
#     )

#     return model, tokenizer 

# if __name__ == "__main__":
#     print("GPU 사용 가능:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print("사용 가능한 GPU:", torch.cuda.device_count())
#         print("현재 GPU:", torch.cuda.current_device())
#         print("GPU 이름:", torch.cuda.get_device_name(0))
    
#     model, tokenizer = load_model_and_tokenizer()
#     print("모델 로딩 완료!") 