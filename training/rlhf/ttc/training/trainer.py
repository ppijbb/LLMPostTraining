from transformers import TrainingArguments
from trl import RewardTrainer # Changed from SFTTrainer
from datasets import Dataset
from models.reward_model import RewardModel
from config.config import ttc_config

class RewardModelTrainer:
    def __init__(self, reward_model: RewardModel, train_dataset: Dataset, eval_dataset: Dataset = None):
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.training_args = TrainingArguments(
            output_dir=ttc_config.REWARD_MODEL_OUTPUT_DIR,
            learning_rate=ttc_config.LEARNING_RATE,
            num_train_epochs=ttc_config.NUM_EPOCHS,
            per_device_train_batch_size=ttc_config.BATCH_SIZE,
            gradient_accumulation_steps=ttc_config.GRADIENT_ACCUMULATION_STEPS,
            save_steps=500, # Example save steps
            logging_steps=100, # Example logging steps
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            fp16=True, # Use mixed precision for faster training
            report_to="none", # Disable reporting to external services like wandb for simplicity
            remove_unused_columns=False, # Important for RewardTrainer which expects specific columns
        )

        # Use RewardTrainer for preference-based reward model training
        self.trainer = RewardTrainer(
            model=self.reward_model.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.reward_model.tokenizer,
            # RewardTrainer expects 'prompt', 'chosen', 'rejected' columns in the dataset
            # You might need to preprocess your dataset accordingly in data_loader.py
        )

    def train(self):
        self.trainer.train()
        self.reward_model.model.save_pretrained(ttc_config.REWARD_MODEL_OUTPUT_DIR)
        self.reward_model.tokenizer.save_pretrained(ttc_config.REWARD_MODEL_OUTPUT_DIR) 