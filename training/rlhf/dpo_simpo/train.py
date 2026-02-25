import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer  # DPOTrainer 사용
from trl import CPOConfig,CPOTrainer  # CPOTrainer 사용
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from data_module import get_dataset
# from unsloth import FastLanguageModel
from accelerate import PartialState

from callbacks import TrainerDebugCallback

# training method 설정
rlhf_method = os.environ.get("RLHF_METHOD", "DPO")

# 모델과 토크나이저 불러오기
# model_id = "AIChenKai/TinyLlama-1.1B-Chat-v1.0-x2-MoE"
model_id = "Gunulhona/Gemma-System-9B"
deepspeed_config = "ds_config.json"

# wandb 설정
os.environ["WANDB_PROJECT"]=f"{model_id.split('/')[1]}-{rlhf_method}"

# save your trained model checkpoint to wandb
# os.environ["WANDB_LOG_MODEL"]="checkpoint"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="0"

# model_id = "microsoft/Phi-3.5-mini-instruct"
max_seq_len = 65536 # if rope scaling is used, max_seq_len should be 65536
batch_per_device = 1
max_epochs = 200
lr = 5e-7 # default 1e-6

print(f'''
--------------------
Model ID: {model_id}
Method : {os.environ.get("RLHF_METHOD")} (default: DPO)
--------------------
''')

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=False,
        #     load_in_8bit=False,
        #     # llm_int8_threshold=6.0,
        #     # llm_int8_has_fp16_weight=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        #     # bnb_4bit_quant_storage=torch.uint8,
        #     bnb_4bit_use_double_quant=True,
        # ),
        device_map={
            "": PartialState().process_index
            # "": torch.cuda.current_device()
        },
        low_cpu_mem_usage=True,
        use_flash_attention_2=False,
        attn_implementation="eager",
        return_dict=True)
tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True)
model.config.use_cache = False
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_id,
#     trust_remote_code=True,
#     use_cache=True, # Use cache for faster decoding
#     max_seq_length=max_seq_len,
#     dtype=torch.bfloat16, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
#     load_in_4bit=False, # Use 4bit quantization to reduce memory usage. Can be False.
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
#     # attn_implementation="flash_attention_2", 
# )

lora_targets=[
    'q_proj',
    'k_proj', 
    'v_proj',
    'o_proj',
    'gate_proj',
    'down_proj',
    'up_proj',
    # "embed_tokens",
    # 'lm_head'
    ]

# 참조 모델 불러오기 (필요에 따라 수정)
# ref_model = model


# Lora를 기본 모델에 적용
peft_config = LoraConfig( # Lora 설정 정의
    use_mora=True,  # Mora Config
    mora_type=6,
    inference_mode=False,
    target_modules=lora_targets,
    r=256,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    # init_lora_weights="gaussian",
    task_type="CAUSAL_LM")

# peft_config = LoHaConfig(
#     target_modules=lora_targets,
#     r=8,
#     alpha=32,
#     task_type="CAUSAL_LM"
# )
model = get_peft_model(
    model=model,
    mixed=False,
    peft_config=peft_config)

# model = FastLanguageModel.get_peft_model(
#     model=model,
#     r=16,
#     max_seq_length=max_seq_len,
#     target_modules=lora_targets,
#     lora_alpha=16,
#     lora_dropout=0, # Dropout = 0 is currently optimized
#     bias="none",    # Bias = "none" is currently optimized
#     use_gradient_checkpointing=True,
#     random_state=3407,
# )
match rlhf_method:
    case "DPO":
        # DPO 설정 정의
        dpo_config = DPOConfig(
            beta=0.01,
            # loss_type="ipo",
            # max_prompt_length=256,
            # max_steps=1000,
            max_target_length=max_seq_len,
            max_length=max_seq_len,
            learning_rate=lr,
            num_train_epochs=max_epochs,
          # trainer options 
            fp16=False,
            bf16=True,
            tf32=False,
            eval_strategy="steps",
            # fp16_full_eval=True,
            # bf16_full_eval=False,
            output_dir="/raid/conan/llm_training/dpo_output",
            # optim="paged_adamw_8bit", # paged_adamw_8bit adamw_bnb_8bit adamw_8bit adamw_hf
            gradient_checkpointing=True,
            logging_steps=20,
            gradient_accumulation_steps=16,
            generate_during_eval=True,
            dataset_num_proc=8,
            report_to="wandb",
            deepspeed=deepspeed_config,
            use_legacy_prediction_loop=False,
            per_device_eval_batch_size=batch_per_device,
            per_device_train_batch_size=batch_per_device,
            per_gpu_eval_batch_size=batch_per_device,
            per_gpu_train_batch_size=batch_per_device,
            lr_scheduler_type="cosine_with_restarts",  #linear, polynomial, reduce_on_plateau, cosine_with_restarts
        )

        # BitsandBytes Paged AdamW Optimizer 설정
        # optimizer_class = bnb.optim.PagedAdamW
        # optimizer_kwargs = {
        #     "lr": dpo_config.learning_rate
        #     }

        dataset = get_dataset(
            dataset_name="Gunulhona/open_dpo_merged",
            tokenizer=tokenizer)

        # DPOTrainer 초기화 (Deepspeed, PagedAdamW 적용)
        rlhf_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            # data_callator=,
            train_dataset=dataset["train"],  # 학습 데이터셋
            eval_dataset=dataset["test"].select(range(50)),  # 학습 데이터셋
            tokenizer=tokenizer,
            # processing_class=tokenizer,
            peft_config=peft_config,
            # optimizers=(bnb.optim.PagedAdamW, {"lr": 3e-5}),
            callbacks=[TrainerDebugCallback()]  # 여러 콜백을 리스트로 전달 가능
        )

    case "SIMPO":
        # CPO 설정 정의
        cpo_config = CPOConfig(
            beta=10.0,
            loss_type="simpo", # SimPO Loss
            simpo_gamma=0.5,
            cpo_alpha=0.0, # SimPO 학습시 0 으로, CPO-SimPO 학습시 0 이상으로 설정
            # max_prompt_length=256,
            # max_steps=1000,
            learning_rate=lr,
            num_train_epochs=max_epochs,            
          # trainer options 
            max_length=max_seq_len,
            fp16=False,
            bf16=True,
            tf32=False,
            # fp16_full_eval=True,
            # bf16_full_eval=False,
            output_dir="/raid/conan/llm_training/cpo_output",
            # optim="paged_adamw_8bit", # paged_adamw_8bit adamw_bnb_8bit adamw_8bit adamw_hf
            gradient_checkpointing=True,
            eval_strategy="steps",
            logging_steps=20,
            gradient_accumulation_steps=16,
            generate_during_eval=True,
            dataset_num_proc=8,
            report_to="wandb",
            deepspeed=deepspeed_config,
            use_legacy_prediction_loop=False,
            per_device_eval_batch_size=batch_per_device,
            per_device_train_batch_size=batch_per_device,
            per_gpu_eval_batch_size=batch_per_device,
            per_gpu_train_batch_size=batch_per_device,
            lr_scheduler_type="cosine_with_restarts",  #linear, polynomial, reduce_on_plateau, cosine_with_restarts
        )

        # BitsandBytes Paged AdamW Optimizer 설정
        # optimizer_class = bnb.optim.PagedAdamW
        # optimizer_kwargs = {
        #     "lr": dpo_config.learning_rate
        #     }

        dataset = get_dataset(
            dataset_name="Gunulhona/open_dpo_merged",
            tokenizer=tokenizer)

        # CPOTrainer 초기화 (Deepspeed, PagedAdamW 적용)
        rlhf_trainer = CPOTrainer(
            model=model,
            # ref_model=None,
            args=cpo_config,
            # data_callator=,
            train_dataset=dataset["train"],  # 학습 데이터셋
            eval_dataset=dataset["test"].select(range(50)),  # 학습 데이터셋
            # tokenizer=tokenizer,
            processing_class=tokenizer,
            peft_config=peft_config,
            # optimizers=(bnb.optim.PagedAdamW, {"lr": 3e-5}),
            callbacks=[TrainerDebugCallback()]  # 여러 콜백을 리스트로 전달 가능
        )


# DPO를 사용하여 모델 학습
if __name__ == "__main__":
    # with torch.autocast("cuda"): 
        rlhf_trainer.train(resume_from_checkpoint=True)
