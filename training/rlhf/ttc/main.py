import sys
import os
# Add the current directory to Python path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import ttc_config
from models.llm_model import LLMModel
from models.reward_model import RewardModel
from models.teacher_model import TeacherModel
from training.trainer import RewardModelTrainer
from training.rlt_trainer import RLTTrainer
from inference.inference_engine import TTCInferenceEngine
from inference.rlt_inference_engine import RLTInferenceEngine
from utils.data_loader import DataLoader
from utils.rlt_data_loader import RLTDataLoader
import argparse
from typing import Optional

def main():
    parser = argparse.ArgumentParser(
        description="Run RLT (Reinforcement Learning Teachers) Test-Time Compute operations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train RLT with GitHub recommended dataset (Bespoke-Stratos-17k)
  python main.py --mode train_rlt
  
  # Train RLT with custom dataset
  python main.py --mode train_rlt --dataset_path /path/to/custom/data.jsonl
  
  # Train RLT with different HuggingFace dataset
  python main.py --mode train_rlt --dataset_name microsoft/DialoGPT-medium
  
  # Run RLT inference
  python main.py --mode rlt_inference --prompt "What is 2+2?" --strategy adaptive
  
  # Run RLT inference with specific scaling
  python main.py --mode rlt_inference --prompt "Solve: 3x + 5 = 20" --strategy fixed_scaling --scaling_factor 2.0
        """
    )
    parser.add_argument("--mode", type=str, 
                        choices=["train_reward", "train_rlt", "train_sft", "train_rl", "inference", "rlt_inference"], 
                        required=True,
                        help="Mode to run: 'train_reward' for reward model, 'train_rlt' for full RLT pipeline, 'train_sft' for SFT only, 'train_rl' for RL only, 'inference' for original TTC, 'rlt_inference' for RLT inference.")
    parser.add_argument("--prompt", type=str, help="Prompt/question for inference mode.")
    parser.add_argument("--strategy", type=str, default="adaptive", 
                        choices=["adaptive", "best_of_n", "majority_voting", "fixed_scaling"],
                        help="Inference strategy for RLT inference.")
    parser.add_argument("--scaling_factor", type=float, default=1.0,
                        help="Scaling factor for test-time compute.")
    parser.add_argument("--target_quality", type=float, default=0.8,
                        help="Target quality for adaptive scaling.")
    parser.add_argument("--dataset_path", type=str, help="Path to custom dataset.")
    parser.add_argument("--dataset_name", type=str, default="bespokelabs/Bespoke-Stratos-17k", 
                        help="HuggingFace dataset name (default: bespokelabs/Bespoke-Stratos-17k from RLT paper)")
    parser.add_argument("--model_path", type=str, help="Path to pre-trained model.")
    args = parser.parse_args()

    # Initialize models based on mode
    teacher_model: Optional[TeacherModel] = None
    llm_model: Optional[LLMModel] = None
    reward_model: Optional[RewardModel] = None

    if args.mode in ["train_rlt", "train_sft", "train_rl", "rlt_inference"]:
        # RLT modes - use teacher model
        print(f"Loading Teacher model: {ttc_config.TEACHER_MODEL_NAME}...")
        teacher_model = TeacherModel(
            model_name=args.model_path or ttc_config.TEACHER_MODEL_NAME, 
            quantization_config=ttc_config.QUANTIZATION_CONFIG
        )
        print("Teacher model loaded.")
        
        # Load reward model for RL phase
        if args.mode in ["train_rl", "train_rlt"]:
            print(f"Loading Reward model: {ttc_config.REWARD_MODEL_NAME}...")
            reward_model = RewardModel(model_name=ttc_config.REWARD_MODEL_NAME)
            print("Reward model loaded.")
    
    else:
        # Original TTC modes
        print(f"Loading LLM model: {ttc_config.LLM_MODEL_NAME}...")
        llm_model = LLMModel(model_name=ttc_config.LLM_MODEL_NAME, quantization_config=ttc_config.QUANTIZATION_CONFIG)
        print("LLM model loaded.")

        print(f"Loading Reward model: {ttc_config.REWARD_MODEL_NAME}...")
        reward_model = RewardModel(model_name=ttc_config.REWARD_MODEL_NAME)
        print("Reward model loaded.")

    # Training modes
    if args.mode == "train_reward":
        print("Starting Reward Model Training...")
        if reward_model is None:
            print("Error: Reward model is required for reward training")
            return
        
        # Load dataset for reward model training - use GitHub recommended dataset by default
        if args.dataset_path:
            print(f"Loading custom dataset from: {args.dataset_path}")
            data_loader = DataLoader(args.dataset_path)
        else:
            print(f"Loading GitHub recommended dataset for reward training: {args.dataset_name}")
            print("This is the Bespoke-Stratos-17k dataset used in the RLT paper")
            # Use RLTDataLoader for the reasoning dataset
            rlt_data_loader = RLTDataLoader(dataset_name=args.dataset_name)
            dataset = rlt_data_loader.load_reasoning_dataset()
            
            # Convert reasoning dataset to reward model training format
            from datasets import Dataset
            reward_data = []
            
            # Ensure dataset is iterable
            if hasattr(dataset, '__iter__'):
                dataset_list = list(dataset)
            else:
                dataset_list = dataset
            
            for i, example in enumerate(dataset_list):
                # Handle both dict and object access
                if isinstance(example, dict):
                    question = example.get('question', '')
                    solution = example.get('solution', '')
                    reasoning_trace = example.get('reasoning_trace', '')
                else:
                    question = getattr(example, 'question', '')
                    solution = getattr(example, 'solution', '')
                    reasoning_trace = getattr(example, 'reasoning_trace', '')
                
                # Positive example (good reasoning)
                positive_text = f"Question: {question}\n<think>{reasoning_trace}</think>\n<solution>{solution}</solution>"
                reward_data.append({"text": positive_text, "label": 1.0})
                
                # Negative example (poor reasoning) - create a simple negative version
                negative_reasoning = "I don't know how to solve this." if reasoning_trace else "No reasoning provided."
                negative_text = f"Question: {question}\n<think>{negative_reasoning}</think>\n<solution>I cannot solve this.</solution>"
                reward_data.append({"text": negative_text, "label": 0.0})
            
            train_dataset = Dataset.from_list(reward_data)
            print(f"Created reward training dataset with {len(train_dataset)} examples")
        
        # If using custom dataset path, load with original DataLoader
        if args.dataset_path:
            train_dataset = data_loader.load_for_reward_model_training()
            
            # Ensure train_dataset is a Dataset, not DatasetDict
            from datasets import DatasetDict, Dataset
            if isinstance(train_dataset, DatasetDict):
                if 'train' in train_dataset:
                    train_dataset = train_dataset['train']
                else:
                    # Take the first available split
                    train_dataset = list(train_dataset.values())[0]
            
            # Ensure we have a proper Dataset type
            if not isinstance(train_dataset, Dataset):
                print(f"Warning: train_dataset is not a Dataset type: {type(train_dataset)}")
                # Try to convert if it's an iterable dataset
                try:
                    if hasattr(train_dataset, '__iter__'):
                        # Convert iterable to list and then to Dataset
                        data_list = list(train_dataset)
                        train_dataset = Dataset.from_list(data_list)
                    else:
                        print("Error: Cannot convert train_dataset to Dataset type")
                        return
                except Exception as e:
                    print(f"Error converting dataset: {e}")
                    return
            
        trainer = RewardModelTrainer(reward_model=reward_model, train_dataset=train_dataset)
        trainer.train()
        print("Reward Model Training Completed and Model Saved.")

    elif args.mode in ["train_rlt", "train_sft", "train_rl"]:
        print(f"Starting RLT Training: {args.mode}...")
        
        if teacher_model is None:
            print("Error: Teacher model is required for RLT training")
            return
        
        # Load reasoning dataset - use GitHub recommended dataset by default
        if args.dataset_path:
            print(f"Loading custom dataset from: {args.dataset_path}")
            rlt_data_loader = RLTDataLoader(data_path=args.dataset_path)
        else:
            print(f"Loading GitHub recommended dataset: {args.dataset_name}")
            print("This is the Bespoke-Stratos-17k dataset used in the RLT paper")
            rlt_data_loader = RLTDataLoader(dataset_name=args.dataset_name)
        
        dataset = rlt_data_loader.load_reasoning_dataset()
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Validate dataset format
        print(f"Dataset columns: {dataset.column_names}")
        if not rlt_data_loader.validate_dataset_format(dataset):
            print("Dataset format validation failed. Exiting.")
            return
        
        # Create train/eval split
        train_dataset, eval_dataset = rlt_data_loader.create_train_eval_split(dataset)
        
        # Initialize RLT trainer
        rlt_trainer = RLTTrainer(teacher_model=teacher_model, reward_model=reward_model)
        
        if args.mode == "train_sft":
            # SFT phase only
            sft_model_path = rlt_trainer.train_sft_phase(train_dataset, eval_dataset)
            print(f"SFT training completed. Model saved to {sft_model_path}")
            
        elif args.mode == "train_rl":
            # RL phase only (requires pre-trained SFT model)
            if not args.model_path:
                print("Error: --model_path is required for RL-only training (should point to SFT model)")
                return
            rl_model_path = rlt_trainer.train_rl_phase(train_dataset, eval_dataset)
            print(f"RL training completed. Model saved to {rl_model_path}")
            
        elif args.mode == "train_rlt":
            # Full RLT pipeline
            final_model_path = rlt_trainer.train_full_pipeline(train_dataset, eval_dataset)
            print(f"Full RLT training completed. Model saved to {final_model_path}")

    # Inference modes
    elif args.mode == "inference":
        if not args.prompt:
            print("Error: --prompt is required for inference mode.")
            return

        if llm_model is None or reward_model is None:
            print("Error: LLM model and reward model are required for inference")
            return

        print(f"Running Original TTC Inference...")
        inference_engine = TTCInferenceEngine(llm_model, reward_model)
        try:
            response, score = inference_engine.run_inference(args.prompt, "best_of_n")
            print("\n--- Original TTC Inference Result ---")
            print(f"Prompt: {args.prompt}")
            print(f"Response: {response}")
            print(f"Score: {score:.4f}")
            print("--------------------------------------")
        except ValueError as e:
            print(f"Inference Error: {e}")

    elif args.mode == "rlt_inference":
        if not args.prompt:
            print("Error: --prompt is required for RLT inference mode.")
            return

        if teacher_model is None:
            print("Error: Teacher model is required for RLT inference")
            return

        print(f"Running RLT Inference with strategy: {args.strategy}")
        rlt_inference_engine = RLTInferenceEngine(teacher_model, reward_model)
        
        try:
            # Prepare inference arguments
            inference_kwargs = {}
            if args.strategy == "adaptive":
                inference_kwargs['target_quality'] = args.target_quality
            elif args.strategy == "fixed_scaling":
                inference_kwargs['scaling_factor'] = args.scaling_factor
            elif args.strategy == "best_of_n":
                inference_kwargs['n'] = ttc_config.BEST_OF_N
                inference_kwargs['scaling_factor'] = args.scaling_factor
            elif args.strategy == "majority_voting":
                inference_kwargs['num_votes'] = 5
                inference_kwargs['scaling_factor'] = args.scaling_factor
            
            result = rlt_inference_engine.run_inference(args.prompt, args.strategy, **inference_kwargs)
            
            print("\n--- RLT Inference Result ---")
            print(f"Question: {result['question']}")
            print(f"Reasoning Trace: {result['reasoning_trace']}")
            print(f"Solution: {result['solution']}")
            print(f"Quality Score: {result['quality_score']:.4f}")
            print(f"Scaling Factor: {result['scaling_factor']}")
            print(f"Tokens Used: {result['tokens_used']}")
            
            if 'adaptive_scaling_info' in result:
                info = result['adaptive_scaling_info']
                print(f"Adaptive Scaling Info:")
                print(f"  Target Quality: {info['target_quality']}")
                print(f"  Scaling Factors Tried: {info['scaling_factors_tried']}")
                print(f"  Quality Achieved: {info['quality_achieved']:.4f}")
                if info.get('target_not_reached'):
                    print("  Target quality not reached with available scaling factors")
            
            if 'majority_voting_info' in result:
                info = result['majority_voting_info']
                print(f"Majority Voting Info:")
                print(f"  Number of Votes: {info['num_votes']}")
                print(f"  Most Common Solution: {info['most_common_solution']}")
            
            print("----------------------------")
            
        except Exception as e:
            print(f"RLT Inference Error: {e}")

if __name__ == "__main__":
    main() 