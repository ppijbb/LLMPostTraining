import sys
import os
# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset, DatasetDict
import json
from typing import Dict, List, Optional, Union
from config.config import ttc_config

class RLTDataLoader:
    """
    Data loader for RLT training pipeline.
    Handles reasoning datasets and prepares data for SFT and RL phases.
    """
    
    def __init__(self, dataset_name: Optional[str] = None, data_path: Optional[str] = None):
        self.dataset_name = dataset_name or ttc_config.REASONING_DATASET
        self.data_path = data_path
    
    def load_reasoning_dataset(self, split: str = "train") -> Dataset:
        """Load reasoning dataset from Hugging Face or local path."""
        try:
            if self.data_path:
                # Load from local path
                dataset = load_dataset("json", data_files=self.data_path, split=split)
            else:
                # Load from Hugging Face
                dataset = load_dataset(self.dataset_name, split=split)
            
            # Ensure we return a Dataset, not DatasetDict
            if isinstance(dataset, DatasetDict):
                if split in dataset:
                    dataset = dataset[split]
                else:
                    # Take the first available split
                    dataset = list(dataset.values())[0]
            
            print(f"Successfully loaded dataset: {len(dataset)} samples")
            
            # Convert Bespoke-Stratos format to RLT format if needed
            if self.dataset_name == "bespokelabs/Bespoke-Stratos-17k":
                dataset = self._convert_bespoke_format(dataset)
            
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Return a small sample dataset for testing
            return self._create_sample_dataset()
    
    def _convert_bespoke_format(self, dataset: Dataset) -> Dataset:
        """Convert Bespoke-Stratos format to RLT format."""
        import re
        
        converted_data = []
        
        for example in dataset:
            conversations = example.get('conversations', [])
            if len(conversations) < 2:
                continue
                
            # Extract user question (first message)
            user_msg = conversations[0]
            if user_msg.get('from') != 'user':
                continue
            question = user_msg.get('value', '')
            
            # Extract assistant response (second message)
            assistant_msg = conversations[1]
            if assistant_msg.get('from') != 'assistant':
                continue
            response = assistant_msg.get('value', '')
            
            # Extract reasoning trace and solution from response
            reasoning_match = re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', response, re.DOTALL)
            solution_match = re.search(r'<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>', response, re.DOTALL)
            
            reasoning_trace = reasoning_match.group(1).strip() if reasoning_match else ""
            solution = solution_match.group(1).strip() if solution_match else ""
            
            # Clean up the solution (remove LaTeX formatting)
            solution = re.sub(r'\\boxed\{([^}]+)\}', r'\1', solution)  # Remove \boxed{}
            solution = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', solution)  # Remove other LaTeX commands
            
            if question and solution:
                converted_data.append({
                    "question": question,
                    "solution": solution,
                    "reasoning_trace": reasoning_trace
                })
        
        print(f"Converted {len(converted_data)} samples from Bespoke-Stratos format")
        return Dataset.from_list(converted_data)
    
    def _create_sample_dataset(self) -> Dataset:
        """Create a sample dataset for testing purposes."""
        sample_data = [
            {
                "question": "What is 2 + 2?",
                "solution": "4",
                "reasoning_trace": "Let me think about this step by step. I need to add 2 and 2 together. 2 represents two units, and when I add another 2 units, I get 4 units total. Therefore, 2 + 2 = 4."
            },
            {
                "question": "If a triangle has angles of 30°, 60°, and 90°, what type of triangle is it?",
                "solution": "Right triangle",
                "reasoning_trace": "Let me analyze the angles: 30° + 60° + 90° = 180°, which is correct for a triangle. Since one angle is exactly 90°, this is a right triangle. The other two angles (30° and 60°) are acute angles."
            },
            {
                "question": "Solve for x: 3x + 5 = 20",
                "solution": "x = 5",
                "reasoning_trace": "I need to solve this linear equation step by step. First, I'll subtract 5 from both sides: 3x + 5 - 5 = 20 - 5, which gives me 3x = 15. Then I'll divide both sides by 3: 3x/3 = 15/3, which gives me x = 5."
            }
        ]
        
        return Dataset.from_list(sample_data)
    
    def prepare_sft_data(self, dataset: Dataset) -> Dataset:
        """Prepare data for SFT training with reasoning format."""
        def format_for_sft(example):
            # Ensure required fields exist
            question = example.get('question', '')
            solution = example.get('solution', '')
            reasoning_trace = example.get('reasoning_trace', '')
            
            # Format with reasoning tags
            formatted_text = (
                f"Question: {question}\n\n"
                f"<think>{reasoning_trace}</think>\n"
                f"<solution>{solution}</solution>"
            )
            
            return {"text": formatted_text}
        
        return dataset.map(format_for_sft)
    
    def prepare_rl_data(self, dataset: Dataset, teacher_model=None) -> Dataset:
        """Prepare data for RL training with chosen/rejected pairs."""
        def create_rl_pairs(example):
            question = example.get('question', '')
            
            if teacher_model:
                # Generate multiple reasoning traces using teacher model
                traces = teacher_model.generate_multiple_traces(question, num_traces=3)
                
                # Evaluate quality of each trace
                scored_traces = []
                for trace in traces:
                    quality_score = teacher_model.evaluate_reasoning_quality(
                        trace['reasoning_trace']
                    )
                    scored_traces.append((trace, quality_score))
                
                # Sort by quality score
                scored_traces.sort(key=lambda x: x[1], reverse=True)
                
                if len(scored_traces) >= 2:
                    chosen_trace = scored_traces[0][0]
                    rejected_trace = scored_traces[1][0]
                    
                    chosen_text = (
                        f"Question: {question}\n"
                        f"<think>{chosen_trace['reasoning_trace']}</think>\n"
                        f"<solution>{chosen_trace['solution']}</solution>"
                    )
                    
                    rejected_text = (
                        f"Question: {question}\n"
                        f"<think>{rejected_trace['reasoning_trace']}</think>\n"
                        f"<solution>{rejected_trace['solution']}</solution>"
                    )
                    
                    return {
                        "prompt": question,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }
            
            # Fallback: use original data if teacher model not available
            if 'reasoning_trace' in example and 'solution' in example:
                # Create a simple pair with the original and a modified version
                original_text = (
                    f"Question: {question}\n"
                    f"<think>{example['reasoning_trace']}</think>\n"
                    f"<solution>{example['solution']}</solution>"
                )
                
                # Create a slightly modified version as rejected
                modified_reasoning = example['reasoning_trace'] + " (This reasoning is less clear.)"
                rejected_text = (
                    f"Question: {question}\n"
                    f"<think>{modified_reasoning}</think>\n"
                    f"<solution>{example['solution']}</solution>"
                )
                
                return {
                    "prompt": question,
                    "chosen": original_text,
                    "rejected": rejected_text
                }
            
            return None
        
        # Apply the function and filter out None values
        rl_dataset = dataset.map(create_rl_pairs).filter(lambda x: x is not None)
        return rl_dataset
    
    def load_evaluation_datasets(self) -> Dict[str, Dataset]:
        """Load evaluation datasets for testing."""
        eval_datasets = {}
        
        for dataset_name in ttc_config.EVAL_DATASETS:
            try:
                # Try to load from Hugging Face
                dataset = load_dataset(dataset_name, split="test")
                
                # Ensure we get a Dataset, not DatasetDict
                if isinstance(dataset, DatasetDict):
                    if "test" in dataset:
                        dataset = dataset["test"]
                    else:
                        dataset = list(dataset.values())[0]
                
                eval_datasets[dataset_name] = dataset
                print(f"Loaded evaluation dataset: {dataset_name} ({len(dataset)} samples)")
            except Exception as e:
                print(f"Could not load {dataset_name}: {e}")
                # Create a small sample for testing
                eval_datasets[dataset_name] = self._create_sample_dataset()
        
        return eval_datasets
    
    def save_processed_data(self, dataset: Dataset, output_path: str):
        """Save processed dataset to file."""
        try:
            dataset.to_json(output_path)
            print(f"Saved processed dataset to {output_path}")
        except Exception as e:
            print(f"Error saving dataset: {e}")
    
    def load_processed_data(self, input_path: str) -> Dataset:
        """Load processed dataset from file."""
        try:
            dataset = load_dataset("json", data_files=input_path, split="train")
            
            # Ensure we return a Dataset, not DatasetDict
            if isinstance(dataset, DatasetDict):
                if "train" in dataset:
                    dataset = dataset["train"]
                else:
                    dataset = list(dataset.values())[0]
            
            print(f"Loaded processed dataset from {input_path} ({len(dataset)} samples)")
            return dataset
        except Exception as e:
            print(f"Error loading processed dataset: {e}")
            return self._create_sample_dataset()
    
    def validate_dataset_format(self, dataset: Dataset) -> bool:
        """Validate that dataset has required format for RLT training."""
        required_columns = ['question', 'solution']
        
        if not all(col in dataset.column_names for col in required_columns):
            print(f"Dataset missing required columns: {required_columns}")
            return False
        
        # Check sample data
        sample = dataset[0]
        if not sample.get('question') or not sample.get('solution'):
            print("Sample data missing question or solution")
            return False
        
        print("Dataset format validation passed")
        return True
    
    def create_train_eval_split(self, dataset: Dataset, eval_ratio: float = 0.1) -> tuple:
        """Create train/eval split from dataset."""
        total_size = len(dataset)
        eval_size = int(total_size * eval_ratio)
        train_size = total_size - eval_size
        
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, total_size))
        
        print(f"Created train/eval split: {len(train_dataset)}/{len(eval_dataset)}")
        return train_dataset, eval_dataset 