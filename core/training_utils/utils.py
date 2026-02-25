import os
import json
from typing import Dict, Any

def format_parameters(number):
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f} B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f} M"
    else:
        return str(number)



def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Config loaded from: {config_path}")
    return config


def setup_deepspeed_environment():
    """Setup environment variables for DeepSpeed optimization"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Enable DeepSpeed optimizations
    if "DEEPSPEED_ZERO_INIT" not in os.environ:
        os.environ["DEEPSPEED_ZERO_INIT"] = "1"
    
    print("DeepSpeed environment variables set")
