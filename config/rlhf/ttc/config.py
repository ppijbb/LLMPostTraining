import os

class TTCConfig:
    # Model Configuration
    LLM_MODEL_NAME = "google/gemma-3n-E4B-it"  # Base model for teacher training
    STUDENT_MODEL_NAME = "google/gemma-3n-E4B-it"  # Student model for distillation
    REWARD_MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"  # Placeholder for reward model
    QUANTIZATION_CONFIG = None  # Example: bitsandbytes.BitsAndBytesConfig(...)
    
    # RLT Teacher Configuration
    TEACHER_MODEL_NAME = "google/gemma-3n-E4B-it"  # Teacher model for reasoning
    TEACHER_OUTPUT_DIR = "models/teacher_checkpoints"
    STUDENT_OUTPUT_DIR = "models/student_checkpoints"
    
    # Training Configuration
    TRAINING_DATA_PATH = "data/reward_training_data.jsonl"
    # GitHub RLT recommended dataset: https://github.com/SakanaAI/RLT
    REASONING_DATASET = "bespokelabs/Bespoke-Stratos-17k"  # RLT-style reasoning dataset
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 3
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 1
    
    # RLT Training Phases
    SFT_LEARNING_RATE = 5e-6  # Supervised Fine-tuning learning rate
    RL_LEARNING_RATE = 1e-6   # Reinforcement Learning learning rate
    KL_PENALTY_COEFF = 0.1    # KL divergence penalty coefficient
    
    # Test-Time Scaling Configuration
    MAX_NEW_TOKENS = 512      # Increased for reasoning traces
    BEST_OF_N = 5             # For Best-of-N sampling
    MCTS_ITERATIONS = 100     # For Monte Carlo Tree Search
    TEMPERATURE = 0.7
    
    # Budget Forcing (RLT-style test-time scaling)
    MIN_THINKING_TOKENS = 50  # Minimum thinking tokens
    MAX_THINKING_TOKENS = 300 # Maximum thinking tokens
    WAIT_TOKEN = "Wait"       # Token to encourage longer thinking
    
    # Evaluation Configuration
    EVAL_DATASETS = ["AIME24", "MATH500", "GPQA"]
    TENSOR_PARALLELISM = 4    # For 7B models
    
    # Paths
    REWARD_MODEL_OUTPUT_DIR = "models/reward_model_checkpoints"
    LOG_DIR = "logs"
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    
    # Reasoning Format Configuration
    REASONING_TAGS = {
        "think_start": "<think>",
        "think_end": "</think>",
        "solution_start": "<solution>",
        "solution_end": "</solution>"
    }
    
    # System prompts for different reasoning formats
    SYSTEM_PROMPTS = {
        "default": "You are a helpful assistant that solves problems step by step.",
        "math": "You are a math tutor. Solve the problem step by step, showing your reasoning.",
        "physics": "You are a physics tutor. Solve the problem step by step, explaining the concepts."
    }

    def __init__(self):
        # Create necessary directories if they don't exist
        os.makedirs(self.REWARD_MODEL_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.TEACHER_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.STUDENT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

# Instantiate the config for easy import
ttc_config = TTCConfig() 