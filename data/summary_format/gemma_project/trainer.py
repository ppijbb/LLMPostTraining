from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
import os
import wandb  # wandb 추가
from transformers.integrations import WandbCallback


def setup_trainer(model, tokenizer, dataset, run_name=None):
    """
    학습 설정을 구성하는 함수
    """
    #model_name = model.config._name_or_path.split("/")[-1]  # 예: 'unsloth/gemma-3-12b-it' -> 'gemma-3-12b-it'

    # wandb 초기화
    wandb.init(
        project="gemma-finetuning",  # 프로젝트 이름
        name=run_name,     # 실험 이름
        config={
            "model": model.config._name_or_path,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "epochs": 7,
            "optimizer": "adamw_8bit",
            #"max_grad_norm": 1.0,
            #"warmup_steps": 100,     # 추가된 warmup_steps
            "scheduler": "cosine",
        }
    )
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = None,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
#            per_device_train_batch_size = 2,
#            gradient_accumulation_steps = 1,
            warmup_steps = 10,
            num_train_epochs = 10,
#            learning_rate = 1e-4,
            learning_rate = 5e-6,
            max_grad_norm = 1.0,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            max_steps = -1,
            lr_scheduler_type = "cosine",
            seed = 3407,
            report_to = "wandb",  # wandb로 리포트하도록 변경
        ),
        callbacks=[WandbCallback()],  # wandb 콜백 추가
    )
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n", 
        response_part = "<start_of_turn>model\n"
        ) 

    # 응답만 학습하도록 설정
    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part = "<start_of_turn>user\n",
    #     response_part = "<start_of_turn>model\n",
    # )
    
    return trainer
# from trl import SFTTrainer, SFTConfig
# from unsloth.chat_templates import train_on_responses_only
# import os

# def setup_trainer(model, tokenizer, dataset):
#     """
#     학습 설정을 구성하는 함수
#     """
#     trainer = SFTTrainer(
#         model = model,
#         tokenizer = tokenizer,
#         train_dataset = dataset,
#         eval_dataset = None,
#         args = SFTConfig(
#             dataset_text_field = "text",
#             per_device_train_batch_size = 2,
#             gradient_accumulation_steps = 4,
#             warmup_steps = 5,
#             num_train_epochs = 5,
#             learning_rate = 2e-4,
#             logging_steps = 1,
#             optim = "adamw_8bit",
#             weight_decay = 0.01,
#             lr_scheduler_type = "linear",
#             seed = 3407,
#             report_to = "none",
#         ),
#     )

# def setup_trainer(model, tokenizer, dataset):
#     """
#     학습 설정을 구성하는 함수
#     """
#     trainer = SFTTrainer(
#         model = model,
#         tokenizer = tokenizer,
#         train_dataset = dataset,
#         eval_dataset = None,
#         args = SFTConfig(
#             dataset_text_field = "text",
#             per_device_train_batch_size = 4,
#             gradient_accumulation_steps = 8 ,
#             warmup_steps = 100,
#             num_train_epochs = 3,
#             learning_rate = 1e-4,
#             logging_steps = 1,
#             optim = "adamw_8bit",
#             weight_decay = 0.01,
#             lr_scheduler_type = "cosine",
#             seed = 3407,
#             report_to = "none",
#         ),
#     )
    
    
#     # 응답만 학습하도록 설정
#     trainer = train_on_responses_only(
#         trainer,
#         instruction_part = "<start_of_turn>user\n",
#         response_part = "<start_of_turn>model\n",
#     )
    
#     return trainer 

