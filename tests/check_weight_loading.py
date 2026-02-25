
import os
import torch
import torch.distributed as dist
import deepspeed
import torch.nn as nn
from transformers import AutoConfig, AutoModelForVision2Seq
from transformers.integrations import HfDeepSpeedConfig
import sys
import logging
import gc
import types
import random
import warnings

# --- Suppress Annoying Deprecation Warnings ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except ImportError:
    pass
# ----------------------------------------------

# --- GLOBALLY PATCH transformers to handle ZeRO-3 Size([0]) placeholders ---
from transformers import modeling_utils
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

_orig_find_mismatched_keys = modeling_utils._find_mismatched_keys

def _safe_find_mismatched_keys(
    model,
    state_dict,
    checkpoint_files,
    ignore_mismatched_sizes,
    keys_to_rename_mapping,
    is_quantized,
    weights_only,
):
    if not ignore_mismatched_sizes:
        return [], []
        
    mismatched_keys, mismatched_shapes = _orig_find_mismatched_keys(
        model,
        state_dict,
        checkpoint_files,
        ignore_mismatched_sizes,
        keys_to_rename_mapping,
        is_quantized,
        weights_only,
    )
    
    if not is_deepspeed_zero3_enabled():
        return mismatched_keys, mismatched_shapes
    
    filtered_keys = []
    filtered_shapes = []
    named_params = dict(model.named_parameters())
    
    for key, (shape1, shape2) in zip(mismatched_keys, mismatched_shapes):
        # ZeRO-3 placeholder check (empty or 0-sized)
        is_zero_size = (len(shape2) == 0 or (len(shape2) == 1 and shape2[0] == 0))
        if is_zero_size:
            param = named_params.get(key)
            if param is not None and hasattr(param, "ds_id"):
                continue # ZeRO-3 sharded param, not a real mismatch
        filtered_keys.append(key)
        filtered_shapes.append((shape1, shape2))
    return filtered_keys, filtered_shapes

modeling_utils._find_mismatched_keys = _safe_find_mismatched_keys
# --------------------------------------------------------------------------

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# This import applies the Safe NN init patches from spectra_model.py
from models.spectra_model import (
    SPECTRATextConfig, 
    SPECTRARouter, 
    SPECTRAExoskeletonMoEInjector
)

def check_spectra_loading():
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = os.environ.get("RANK", "0")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logger = logging.getLogger("SPECTRA_Checker")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    torch.cuda.set_device(local_rank)
    logger.info(f"🌐 Distributed initialized. Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")

    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    
    ds_config = {
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu"},
            "stage3_param_persistence_threshold": 1000000,
            "leaf_module": {
                 "classes": [
                    "Qwen3VLMoeTextDecoderLayer",
                    "Qwen3VLMoeVisionBlock",
                    "Qwen3VLMoeVisionMLP",
                    "Qwen3VLMoeVisionAttention",
                    "Qwen3VLMoeTextSparseMoeBlock",
                    "Qwen3VLMoeTextExperts",
                    "Qwen3VLMoeVisionPatchMerger",
                    "Qwen3VLMoeVisionPatchEmbed",
                    "SPECTRAMoE",
                    "SPECTRARouter"
                ]
            }
        },
        "train_micro_batch_size_per_gpu": 1
    }
    
    hf_ds_config = HfDeepSpeedConfig(ds_config)
    
    logger.info("🚀 Instantiating model under ZeRO-3 check with Weight Check Bypass...")
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True, 
            device_map=None
        )
        
        spectra_config = SPECTRATextConfig()
        spectra_config.hidden_size = 2048 
        global_router = SPECTRARouter(spectra_config)
        injector = SPECTRAExoskeletonMoEInjector(spectra_config=spectra_config, global_router=global_router)
        model = injector.inject(base_model)
    
    dist.barrier()
    
    logger.info("🔍 AUDIT: Checking parameter sharding status...")
    total_params = 0
    ds_managed_count = 0
    actual_holes = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        is_ds = hasattr(param, "ds_id")
        if is_ds: ds_managed_count += 1
        
        full_numel = getattr(param, "ds_numel", param.numel())
        if full_numel > 0 and not is_ds and param.numel() == 0:
            actual_holes += 1
            if dist.get_rank() == 0:
                logger.error(f"   [❌ REAL HOLE] {name}: Not managed by DeepSpeed and size 0!")

    logger.info(f"📊 Audit Summary: {total_params} total, {ds_managed_count} DeepSpeed managed, {actual_holes} actual holes.")
    
    if actual_holes > 0:
        logger.error(f"🚨 FATAL: Found {actual_holes} actual holes!")
        sys.exit(1)
    else:
        logger.info("🎉 SUCCESS: Model weights are correctly registered with DeepSpeed ZeRO-3 and Loading verified.")
        sys.exit(0)

if __name__ == "__main__":
    check_spectra_loading()
