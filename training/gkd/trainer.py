import torch
from torch import nn
from torch.optim import Adam

from datasets import Dataset, load_dataset
from trl import GKDConfig, GKDTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from .data_module import DistillationDataModule
from .utils import format_parameters

NUM_DUMMY_SAMPLES = 100

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# The model to optimise
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# The teacher model to calculate the KL divergence against
teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

dataset = load_dataset(
    "Gunulhona/open_m_3", 
    split="train", 
    streaming=True
    ).train_test_split(test_size=0.2)

training_args = GKDConfig(output_dir="gkd-model", per_device_train_batch_size=1)
trainer = GKDTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

def train_distillation():
    print(f"Training with {format_parameters(model.num_parameters())} parameters")
    print(f"Teacher model has {format_parameters(teacher_model.num_parameters())} parameters")
    trainer.train()
