from transformers.training_args import TrainingArguments
from trl import SFTTrainer, RewardTrainer
from datasets import Dataset
import torch
from typing import Dict, List, Optional
import os
import sys
# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ttc_config
from models.teacher_model import TeacherModel

class RLTTrainer:
    """
    RLT-style trainer that implements the two-phase training process:
    1. Supervised Fine-tuning (SFT) phase
    2. Reinforcement Learning (RL) phase
    """
    
    def __init__(self, teacher_model: TeacherModel, reward_model=None):
        self.teacher_model = teacher_model
        self.reward_model = reward_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def prepare_sft_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for SFT training with reasoning format."""
        def format_for_sft(example):
            # Format: system_prompt + question + <think> + reasoning + </think> + <solution> + solution + </solution>
            system_prompt = ttc_config.SYSTEM_PROMPTS.get("default", "You are a helpful assistant.")
            
            if "reasoning_trace" in example:
                formatted_text = (
                    f"{system_prompt}\n\nQuestion: {example['question']}\n\n"
                    f"<think>{example['reasoning_trace']}</think>\n"
                    f"<solution>{example['solution']}</solution>"
                )
            else:
                # If no reasoning trace, generate one
                trace_result = self.teacher_model.generate_reasoning_trace(
                    example['question'], 
                    max_tokens=ttc_config.MAX_NEW_TOKENS
                )
                formatted_text = (
                    f"{system_prompt}\n\nQuestion: {example['question']}\n\n"
                    f"<think>{trace_result['reasoning_trace']}</think>\n"
                    f"<solution>{trace_result['solution']}</solution>"
                )
            
            # Return both text and messages format for compatibility
            return {
                "text": formatted_text,
                "messages": [
                    {"role": "user", "content": example['question']},
                    {"role": "assistant", "content": formatted_text}
                ]
            }
        
        return dataset.map(format_for_sft)
    
    def train_sft_phase(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Supervised Fine-tuning phase."""
        print("Starting SFT phase...")
        
        # Prepare datasets
        sft_dataset = self.prepare_sft_dataset(train_dataset)
        sft_eval_dataset = None
        if eval_dataset is not None:
            sft_eval_dataset = self.prepare_sft_dataset(eval_dataset)
        
        # Training arguments for SFT
        training_args = TrainingArguments(
            output_dir=os.path.join(ttc_config.TEACHER_OUTPUT_DIR, "sft"),
            learning_rate=ttc_config.SFT_LEARNING_RATE,
            num_train_epochs=ttc_config.NUM_EPOCHS,
            per_device_train_batch_size=ttc_config.BATCH_SIZE,
            gradient_accumulation_steps=ttc_config.GRADIENT_ACCUMULATION_STEPS,
            save_steps=500,
            logging_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",  # Disable wandb for now
        )
        
        # Initialize SFT trainer
        sft_trainer = SFTTrainer(
            model=self.teacher_model.model,
            processing_class=self.teacher_model.tokenizer,
            train_dataset=sft_dataset,
            eval_dataset=sft_eval_dataset,
            args=training_args,
        )
        
        # Train
        sft_trainer.train()
        
        # Save the SFT model
        sft_output_dir = os.path.join(ttc_config.TEACHER_OUTPUT_DIR, "sft_final")
        self.teacher_model.save_model(sft_output_dir)
        print(f"SFT phase completed. Model saved to {sft_output_dir}")
        
        return sft_output_dir
    
    def prepare_rl_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for RL training with chosen/rejected pairs."""
        def create_rl_pairs(example):
            # Generate multiple reasoning traces for the same question
            traces = self.teacher_model.generate_multiple_traces(
                example['question'], 
                num_traces=3
            )
            
            # Evaluate quality of each trace
            scored_traces = []
            for trace in traces:
                quality_score = self.teacher_model.evaluate_reasoning_quality(
                    trace['reasoning_trace']
                )
                scored_traces.append((trace, quality_score))
            
            # Sort by quality score
            scored_traces.sort(key=lambda x: x[1], reverse=True)
            
            # Create chosen/rejected pairs
            if len(scored_traces) >= 2:
                chosen_trace = scored_traces[0][0]
                rejected_trace = scored_traces[1][0]
                
                chosen_text = (
                    f"Question: {example['question']}\n"
                    f"<think>{chosen_trace['reasoning_trace']}</think>\n"
                    f"<solution>{chosen_trace['solution']}</solution>"
                )
                
                rejected_text = (
                    f"Question: {example['question']}\n"
                    f"<think>{rejected_trace['reasoning_trace']}</think>\n"
                    f"<solution>{rejected_trace['solution']}</solution>"
                )
                
                return {
                    "prompt": example['question'],
                    "chosen": chosen_text,
                    "rejected": rejected_text
                }
            
            return None
        
        # Apply the function and filter out None values
        rl_dataset = dataset.map(create_rl_pairs).filter(lambda x: x is not None)
        return rl_dataset
    
    def train_rl_phase(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Reinforcement Learning phase using RewardTrainer."""
        print("Starting RL phase...")
        
        if self.reward_model is None:
            raise ValueError("Reward model is required for RL phase")
        
        # Prepare dataset
        rl_dataset = self.prepare_rl_dataset(train_dataset)
        
        # Training arguments for RL
        training_args = TrainingArguments(
            output_dir=os.path.join(ttc_config.TEACHER_OUTPUT_DIR, "rl"),
            learning_rate=ttc_config.RL_LEARNING_RATE,
            num_train_epochs=ttc_config.NUM_EPOCHS,
            per_device_train_batch_size=ttc_config.BATCH_SIZE,
            gradient_accumulation_steps=ttc_config.GRADIENT_ACCUMULATION_STEPS,
            save_steps=500,
            logging_steps=100,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",
        )
        
        # Initialize RL trainer
        rl_trainer = RewardTrainer(
            model=self.teacher_model.model,
            processing_class=self.teacher_model.tokenizer,
            train_dataset=rl_dataset,
            eval_dataset=eval_dataset if eval_dataset else None,
            args=training_args,
        )
        
        # Train
        rl_trainer.train()
        
        # Save the RL model
        rl_output_dir = os.path.join(ttc_config.TEACHER_OUTPUT_DIR, "rl_final")
        self.teacher_model.save_model(rl_output_dir)
        print(f"RL phase completed. Model saved to {rl_output_dir}")
        
        return rl_output_dir
    
    def train_full_pipeline(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Run the complete RLT training pipeline."""
        print("Starting RLT training pipeline...")
        
        # Phase 1: SFT
        sft_model_path = self.train_sft_phase(train_dataset, eval_dataset)
        
        # Phase 2: RL (if reward model is available)
        if self.reward_model is not None:
            rl_model_path = self.train_rl_phase(train_dataset, eval_dataset)
            print(f"Full RLT pipeline completed. Final model saved to {rl_model_path}")
            return rl_model_path
        else:
            print("Reward model not provided. Skipping RL phase.")
            print(f"Training completed with SFT only. Model saved to {sft_model_path}")
            return sft_model_path 