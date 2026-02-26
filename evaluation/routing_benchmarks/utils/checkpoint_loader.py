"""Checkpoint loading utilities for Seqorth and baseline models."""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging

logger = logging.getLogger(__name__)


def load_seqorth_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: Optional[str] = "auto",
    **kwargs
) -> tuple[nn.Module, Any]:
    """
    Load Seqorth checkpoint with DeepSpeed Zero-3 support.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load model on
        torch_dtype: Data type for model weights
        load_in_8bit: Whether to use 8-bit quantization
        load_in_4bit: Whether to use 4-bit quantization
        device_map: Device mapping strategy
        **kwargs: Additional arguments for model loading
    
    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading Seqorth checkpoint from {checkpoint_path}")
    
    # Check for DeepSpeed Zero-3 checkpoint
    zero_checkpoint_path = checkpoint_path / "zero_to_fp32.py"
    if zero_checkpoint_path.exists():
        logger.info("Detected DeepSpeed Zero-3 checkpoint, converting...")
        consolidated_path = checkpoint_path / "pytorch_model.bin"
        if not consolidated_path.exists():
            # Run zero_to_fp32 conversion
            import subprocess
            result = subprocess.run(
                ["python", str(zero_checkpoint_path), str(checkpoint_path), str(consolidated_path)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Zero-3 conversion failed: {result.stderr}")
                raise RuntimeError("Failed to convert Zero-3 checkpoint")
        checkpoint_path = consolidated_path
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
        logger.info("Loaded tokenizer successfully")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from checkpoint: {e}")
        logger.info("Attempting to load tokenizer from base model config...")
        config_file = checkpoint_path / "config.json"
        if config_file.exists():
            import json
            with open(config_file) as f:
                config = json.load(f)
            base_model = config.get("_name_or_path", "Qwen/Qwen2.5-1.5B")
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        else:
            raise RuntimeError("Could not load tokenizer")
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            **kwargs
        )
        logger.info(f"Loaded Seqorth model successfully")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    return model, tokenizer


def load_baseline_model(
    model_config: Dict[str, Any],
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: Optional[str] = "auto",
    use_auth_token: bool = False,
    **kwargs
) -> tuple[nn.Module, Any]:
    """
    Load baseline model from HuggingFace Hub.
    
    Args:
        model_config: Model configuration dict from baseline_models.yaml
        cache_dir: Cache directory for downloaded models
        device: Device to load model on
        torch_dtype: Data type for model weights
        load_in_8bit: Whether to use 8-bit quantization
        load_in_4bit: Whether to use 4-bit quantization
        device_map: Device mapping strategy
        use_auth_token: Whether to use HF auth token for gated models
        **kwargs: Additional arguments for model loading
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_config.get("name", "Unknown")
    hf_path = model_config["hf_path"]
    
    # Override with config values if provided
    if "loading" in model_config:
        loading_config = model_config["loading"]
        cache_dir = loading_config.get("cache_dir", cache_dir)
        load_in_8bit = loading_config.get("load_in_8bit", load_in_8bit)
        load_in_4bit = loading_config.get("load_in_4bit", load_in_4bit)
        torch_dtype_str = loading_config.get("torch_dtype", "bfloat16")
        if torch_dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype_str == "float16":
            torch_dtype = torch.float16
        elif torch_dtype_str == "float32":
            torch_dtype = torch.float32
    
    logger.info(f"Loading baseline model: {model_name} from {hf_path}")
    
    # Create cache directory if needed
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    # Get auth token if needed
    auth_token = None
    if use_auth_token:
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            auth_token = token_file.read_text().strip()
        else:
            logger.warning("Auth token requested but not found at ~/.huggingface/token")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=auth_token
        )
        logger.info(f"Loaded tokenizer for {model_name}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")
        raise
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            token=auth_token,
            **kwargs
        )
        logger.info(f"Loaded {model_name} successfully")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        raise
    
    return model, tokenizer


def prepare_model_for_eval(
    model: nn.Module,
    disable_dropout: bool = True,
    disable_layer_norm: bool = False,
) -> nn.Module:
    """
    Prepare model for evaluation.
    
    Args:
        model: Model to prepare
        disable_dropout: Whether to disable dropout layers
        disable_layer_norm: Whether to set layer norm to eval mode
    
    Returns:
        Prepared model
    """
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Disable dropout
    if disable_dropout:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
    
    # Set layer norm to eval mode
    if disable_layer_norm:
        for module in model.modules():
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
                module.eval()
    
    logger.info("Model prepared for evaluation")
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Extract model information for reporting.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with model information
    """
    info = {
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }
    
    # Try to extract MoE-specific info
    try:
        if hasattr(model, "config"):
            config = model.config
            if hasattr(config, "num_experts"):
                info["num_experts"] = config.num_experts
            if hasattr(config, "num_experts_per_tok"):
                info["top_k"] = config.num_experts_per_tok
            if hasattr(config, "num_hidden_layers"):
                info["num_layers"] = config.num_hidden_layers
            if hasattr(config, "hidden_size"):
                info["hidden_size"] = config.hidden_size
    except Exception as e:
        logger.debug(f"Could not extract config info: {e}")
    
    return info


def estimate_active_parameters(model: nn.Module, top_k: int = 2, num_experts: int = 8) -> int:
    """
    Estimate active parameters for MoE models.
    
    For dense models, returns total parameters.
    For MoE models, estimates active params based on routing.
    
    Args:
        model: Model to analyze
        top_k: Number of experts activated per token
        num_experts: Total number of experts
    
    Returns:
        Estimated active parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Try to detect MoE layers
    moe_params = 0
    non_moe_params = 0
    
    for name, module in model.named_modules():
        if "expert" in name.lower() or "moe" in name.lower():
            moe_params += sum(p.numel() for p in module.parameters())
        else:
            # Count parameters in non-expert layers
            if len(list(module.children())) == 0:  # Leaf module
                non_moe_params += sum(p.numel() for p in module.parameters())
    
    if moe_params > 0:
        # MoE model: active = non_moe + (moe * top_k / num_experts)
        active_params = non_moe_params + int(moe_params * top_k / num_experts)
        logger.info(f"MoE model detected: {moe_params:,} expert params, {non_moe_params:,} non-expert params")
        logger.info(f"Active params (top-{top_k}/{num_experts}): {active_params:,}")
    else:
        # Dense model: all parameters active
        active_params = total_params
        logger.info(f"Dense model detected: {active_params:,} active params")
    
    return active_params

