#!/usr/bin/env python3
"""
SPECTRA Light Training Script
Simplified version of train_spectra.py - exactly matching dataset loading and model options.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import logging
import warnings
import time
from typing import Dict, Any, Tuple, Callable

# Add workspace root to path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Expect PYTHONPATH to include repo root (workspace_root)

from transformers import AutoTokenizer, AutoProcessor, AutoConfig, set_seed, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from core.training_utils import (
    setup_dataset, load_config, clear_gpu_memory, 
    ensure_router_parameters_trainable, protect_vision_freeze,
    setup_logging, get_dynamic_lora_target_modules, count_unique_parameters,
    format_parameters
)
from models.spectra_model import SPECTRAForConditionalGeneration, SPECTRAConfig, SPECTRAMoE
from core.optimizers.deepspeed_optimizer_registry import register_custom_optimizers

# --- CRITICAL: Robust Monkey-patches ---
def apply_robust_patches():
    """Apply all robust monkey-patches to handle dtype mismatches and DeepSpeed Stage 2/3 edge cases."""
    print("🛠️ Applying robust monkey-patches...", flush=True)

    # 1. DeepSpeed Gradient Patch (Handles None grad)
    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        if hasattr(DeepSpeedZeroOptimizer, 'reduce_independent_p_g_buckets_and_remove_grads'):
            original_reduce = DeepSpeedZeroOptimizer.reduce_independent_p_g_buckets_and_remove_grads
            def patched_reduce(self, param, i):
                if param.grad is None:
                    param.grad = torch.zeros_like(param, dtype=param.dtype)
                    param.grad.requires_grad = False
                return original_reduce(self, param, i)
            DeepSpeedZeroOptimizer.reduce_independent_p_g_buckets_and_remove_grads = patched_reduce
            print("✅ Applied DeepSpeed ZeRO-2 Gradient Patch", flush=True)
    except Exception as e:
        print(f"⚠️ Failed to apply ZeRO-2 Gradient Patch: {e}", flush=True)

    # 2. DeepSpeed ZeRO-3 Gradient Patch
    try:
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
        mangled_name = '_DeepSpeedZeroOptimizer_Stage3__add_grad_to_ipg_bucket'
        if hasattr(DeepSpeedZeroOptimizer_Stage3, mangled_name):
            original_add_grad = getattr(DeepSpeedZeroOptimizer_Stage3, mangled_name)
            def patched_add_grad(self, param):
                if param.grad is None:
                    param.grad = torch.zeros_like(param, dtype=param.dtype)
                return original_add_grad(self, param)
            setattr(DeepSpeedZeroOptimizer_Stage3, mangled_name, patched_add_grad)
            print("✅ Applied DeepSpeed ZeRO-3 Stage 3 Gradient Patch", flush=True)
    except Exception: pass

    # 3. Disagreement Patch
    try:
        import deepspeed.runtime.zero.partitioned_param_coordinator as ppc
        ppc.assert_ints_same_as_other_ranks = lambda *args, **kwargs: None
        ppc.assert_lst_len_same_as_other_ranks = lambda *args, **kwargs: None
        print("✅ Applied DeepSpeed Disagreement Patch", flush=True)
    except Exception: pass

    # 4. BitsAndBytes + DeepSpeed ZeRO-2 Compatibility Patch (Memory-Efficient)
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnb_func
        import torch
        
        # Patch is_on_gpu to move CPU tensors to GPU efficiently (in-place)
        original_is_on_gpu = bnb_func.is_on_gpu
        
        def patched_is_on_gpu(tensors):
            """Memory-efficient: Move CPU tensors to GPU in-place."""
            if not torch.cuda.is_available():
                return original_is_on_gpu(tensors)
            
            # Move in-place to avoid memory duplication
            for i, tensor in enumerate(tensors):
                if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu':
                    tensors[i] = tensor.cuda(non_blocking=True)
            
            return original_is_on_gpu(tensors)
        
        bnb_func.is_on_gpu = patched_is_on_gpu
        
        # Patch optimizer update_step (memory-efficient)
        for opt_class_name in ['AdamW8bit', 'PagedAdamW8bit', 'Adam8bit', 'PagedAdam8bit']:
            if hasattr(bnb.optim, opt_class_name):
                opt_class = getattr(bnb.optim, opt_class_name)
                if hasattr(opt_class, 'update_step'):
                    original_update_step = opt_class.update_step
                    
                    def make_patched_update_step(original):
                        def patched_update_step(self, p, g, state1, state2, qmap1, qmap2, absmax1, absmax2, *args, **kwargs):
                            """Move tensors to GPU in-place only if on CPU."""
                            if not torch.cuda.is_available():
                                return original(self, p, g, state1, state2, qmap1, qmap2, absmax1, absmax2, *args, **kwargs)
                            
                            # Move in-place with non_blocking for better performance
                            if isinstance(p, torch.Tensor) and p.device.type == 'cpu':
                                p = p.cuda(non_blocking=True)
                            if isinstance(g, torch.Tensor) and g.device.type == 'cpu':
                                g = g.cuda(non_blocking=True)
                            if isinstance(state1, torch.Tensor) and state1.device.type == 'cpu':
                                state1 = state1.cuda(non_blocking=True)
                            if isinstance(state2, torch.Tensor) and state2.device.type == 'cpu':
                                state2 = state2.cuda(non_blocking=True)
                            if isinstance(qmap1, torch.Tensor) and qmap1.device.type == 'cpu':
                                qmap1 = qmap1.cuda(non_blocking=True)
                            if isinstance(qmap2, torch.Tensor) and qmap2.device.type == 'cpu':
                                qmap2 = qmap2.cuda(non_blocking=True)
                            if isinstance(absmax1, torch.Tensor) and absmax1.device.type == 'cpu':
                                absmax1 = absmax1.cuda(non_blocking=True)
                            if isinstance(absmax2, torch.Tensor) and absmax2.device.type == 'cpu':
                                absmax2 = absmax2.cuda(non_blocking=True)
                            
                            return original(self, p, g, state1, state2, qmap1, qmap2, absmax1, absmax2, *args, **kwargs)
                        return patched_update_step
                    
                    opt_class.update_step = make_patched_update_step(original_update_step)
        
        print("✅ Applied BitsAndBytes + DeepSpeed ZeRO-2 Compatibility Patch (Memory-Efficient)", flush=True)
    except Exception as e:
        print(f"⚠️ Failed to apply BitsAndBytes compatibility patch: {e}", flush=True)

# --- CRITICAL: SymmetricSFTTrainer ---
# We force all ranks to see the EXACT SAME data batch by using a unified sampler.
# This prevents rank desync caused by different image resolutions or skips.
class SymmetricSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset=None) -> torch.utils.data.Sampler:
        target_dataset = dataset if dataset is not None else self.train_dataset
        if target_dataset is None:
            return None
        return torch.utils.data.distributed.DistributedSampler(
            target_dataset,
            num_replicas=1, # LIE: force all ranks to see all data
            rank=0,
            seed=self.args.seed
        )
    
    def _get_eval_sampler(self, eval_dataset) -> torch.utils.data.Sampler:
        if eval_dataset is None:
            return None
        return torch.utils.data.distributed.DistributedSampler(
            eval_dataset,
            num_replicas=1,
            rank=0,
            seed=self.args.seed
        )

def main():
    import os
    import torch
    
    # Set PyTorch CUDA allocator config for memory efficiency
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    
    parser = argparse.ArgumentParser(description="SPECTRA Light Training")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args, unknown = parser.parse_known_args()

    # Apply patches and register optimizers
    apply_robust_patches()
    register_custom_optimizers()

    # Load configurations
    config_data = load_config(args.config)
    model_config = config_data["model_config"]
    data_config = config_data["data_config"]
    training_config = config_data["training_config"]

    set_seed(training_config.get("seed", 42))

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    
    logger = setup_logging()
    
    if rank == 0:
        logger.info(f"🚀 Initializing Light Training for {model_config['model_name_or_path']}...")

    # setup tokenizer/processor matching train_spectra.py
    tokenizer_path = model_config.get("tokenizer_name_or_path") or model_config["model_name_or_path"]
    try:
        tokenizer_or_processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_config.get("trust_remote_code", True)
        )
        logger.info("  ✅ AutoProcessor loaded successfully")
    except Exception:
        tokenizer_or_processor = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_config.get("trust_remote_code", True)
        )
        logger.info("  ✅ AutoTokenizer loaded successfully")

    # Dataset setup EXACTLY matching train_spectra.py
    dataset, collate_fn = setup_dataset(data_config, tokenizer_or_processor, logger)
    train_dataset = dataset.get("train", None)
    eval_dataset = dataset.get("test", None)
    
    if eval_dataset is None and train_dataset is not None:
        logger.info("Splitting train dataset for evaluation (10%)...")
        splited = train_dataset.train_test_split(test_size=0.1)
        train_dataset = splited["train"]
        eval_dataset = splited["test"]

    if train_dataset is None or len(train_dataset) == 0:
        raise ValueError("훈련 데이터셋이 비어있습니다! 데이터셋 로딩을 확인하세요.")

    # Setup model configuration
    base_config = AutoConfig.from_pretrained(
        model_config["model_name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", True)
    )
    
    spectra_params = model_config.get("spectra_params", {})
    text_config_dict = base_config.to_dict().get("text_config", base_config.to_dict())
    text_config_dict.update(spectra_params)
    
    config = SPECTRAConfig(
        text_config=text_config_dict,
        vision_config=base_config.to_dict().get("vision_config"),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )
    config.tie_word_embeddings = False # Critical for NVMe/ZeRO-3 stability
    
    # Load model with memory-efficient settings
    model_dtype = torch.bfloat16 if training_config.get("bf16") else torch.float16
    logger.info(f"🤖 Loading SPECTRA model on rank {rank} with {model_dtype}...")
    
    # Memory-efficient model loading for DeepSpeed
    device_map = None
    max_memory = None
    dschf = None  # HfDeepSpeedConfig reference (must keep alive)
    
    if model_config.get("deepspeed_config"):
        # CRITICAL: Use HfDeepSpeedConfig for meta device initialization
        # This prevents OOM by initializing model on meta device, then loading sharded
        ds_config_path = model_config.get("deepspeed_config")
        if isinstance(ds_config_path, str) and os.path.exists(ds_config_path):
            with open(ds_config_path, 'r') as f:
                ds_config_dict = json.load(f)
        elif isinstance(ds_config_path, dict):
            ds_config_dict = ds_config_path.copy()
        else:
            ds_config_dict = {}
        
        # CRITICAL: Calculate train_batch_size BEFORE HfDeepSpeedConfig initialization
        # Trainer calculates: train_batch_size = per_device * grad_accum * world_size
        # Match train_spectra.py: patch "auto" to 1 for zero.Init() validation, then calculate actual value
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_device = training_config.get("per_device_train_batch_size", 1)
        grad_accum = training_config.get("gradient_accumulation_steps", 1)
        
        if ds_config_dict.get("train_batch_size") == "auto":
            calculated_batch = per_device * grad_accum * world_size
            ds_config_dict["train_batch_size"] = calculated_batch
            logger.info(f"  🔧 Calculated train_batch_size: {per_device} * {grad_accum} * {world_size} = {calculated_batch}")
        
        if ds_config_dict.get("train_micro_batch_size_per_gpu") == "auto":
            ds_config_dict["train_micro_batch_size_per_gpu"] = per_device
        
        if ds_config_dict.get("gradient_accumulation_steps") == "auto":
            ds_config_dict["gradient_accumulation_steps"] = grad_accum
        
        # Patch optimizer params if "auto" - use training config values
        if "optimizer" in ds_config_dict and "params" in ds_config_dict["optimizer"]:
            for k in ["lr", "weight_decay"]:
                if ds_config_dict["optimizer"]["params"].get(k) == "auto":
                    if k == "lr":
                        ds_config_dict["optimizer"]["params"][k] = training_config.get("learning_rate", 1e-5)
                    elif k == "weight_decay":
                        ds_config_dict["optimizer"]["params"][k] = training_config.get("weight_decay", 0.0)
        
        # CRITICAL: Save modified config to temporary file for HfDeepSpeedConfig
        import tempfile
        temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(ds_config_dict, temp_config_file, indent=2)
        temp_config_file.close()
        temp_config_path = temp_config_file.name
        logger.info(f"  💾 Saved modified DeepSpeed config to temporary file: {temp_config_path}")
        
        # Update model_config to use temporary config file
        model_config["deepspeed_config"] = temp_config_path
        
        # Check ZeRO stage - only ZeRO-3 supports meta device initialization
        zero_stage = ds_config_dict.get("zero_optimization", {}).get("stage", 0)
        
        if zero_stage == 3:
            # CRITICAL: ZeRO-3 supports meta device initialization via HfDeepSpeedConfig
            # This prevents OOM by initializing model on meta device, then loading sharded
            try:
                from transformers.integrations import HfDeepSpeedConfig
                # Use temp file path so deepspeed_config() returns the modified config
                dschf = HfDeepSpeedConfig(temp_config_path)  # Must keep reference alive
                logger.info("  🚀 HfDeepSpeedConfig initialized - meta device sharded loading enabled (ZeRO-3)")
                device_map = None  # DeepSpeed handles placement
                max_memory = None  # Not needed with meta device init
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to initialize HfDeepSpeedConfig: {e}")
                logger.info("  🔄 Falling back to low_cpu_mem_usage mode...")
                device_map = None
                max_memory = None
        else:
            # ZeRO-2 does NOT support meta device init
            # Load model to CPU first, then DeepSpeed will move to GPU during initialization
            logger.info(f"  ⚡ ZeRO Stage {zero_stage} detected - loading to CPU first (ZeRO-2 doesn't support meta device)")
            logger.info("  💡 Model will be moved to GPU during DeepSpeed initialization")
            device_map = "cpu"  # Load to CPU first to avoid OOM
            max_memory = None
    else:
        # Without DeepSpeed, use auto device_map if multiple GPUs
        if torch.cuda.device_count() > 1:
            device_map = "auto"
            logger.info(f"  🔄 Using auto device_map for {torch.cuda.device_count()} GPUs")
    
    # Clear memory before loading
    if rank == 0:
        clear_gpu_memory(logger)
    
    # Load model with memory-efficient settings
    model = SPECTRAForConditionalGeneration.from_pretrained(
        model_config["model_name_or_path"],
        config=config,
        trust_remote_code=model_config.get("trust_remote_code", True),
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        max_memory=max_memory,
    )
    
    # Keep dschf reference alive (required for HfDeepSpeedConfig)
    if dschf is not None:
        model._dschf = dschf
        # Store temp config path for cleanup later
        if 'temp_config_path' in locals():
            model._temp_ds_config_path = temp_config_path

    # PEFT / LoRA setup EXACTLY matching train_spectra.py
    if model_config.get("use_lora"):
        logger.info("🔍 Applying LoRA...")
        if "lora_target_modules" in model_config and model_config["lora_target_modules"]:
            lora_target_modules = model_config["lora_target_modules"]
            logger.info(f"✅ Using explicit LoRA target modules from config: {lora_target_modules}")
        else:
            lora_target_modules = get_dynamic_lora_target_modules(model, logger)
        
        # Verify any matches exist before applying peft to avoid ValueError
        found_any = False
        all_module_names = [n for n, _ in model.named_modules()]
        for target in lora_target_modules:
            # Suffix match logic similar to PEFT
            if any(name.endswith(target) or f".{target}." in f".{name}." for name in all_module_names):
                found_any = True
                break
        
        if not found_any:
            logger.warning("⚠️ No LoRA target modules found with suffix matching. Trying regex match...")
            # If no suffix matches, it might be the layers.*.mlp patterns.
            import re
            for target in lora_target_modules:
                pattern = target.replace("*", ".*")
                if any(re.search(pattern, name) for name in all_module_names):
                    found_any = True
                    break
        
        if found_any:
            lora_config = LoraConfig(
                r=model_config["lora_r"],
                lora_alpha=model_config["lora_alpha"],
                target_modules=lora_target_modules,
                lora_dropout=model_config.get("lora_dropout", 0.00),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.enable_input_require_grads()
            logger.info("✅ LoRA applied successfully.")
        else:
            logger.error(f"❌ Could not find ANY LoRA target modules in model! Targets: {lora_target_modules}")
            if rank == 0:
                logger.info(f"First 20 module names: {all_module_names[:20]}")
            raise ValueError("LoRA target modules not found")
    
    # Expert training setup matching train_spectra.py (lines 1681-1736)
    if model_config.get("train_all_experts", True):
        logger.info("🔍 Enabling expert parameter training for ALL experts...")
        expert_params_made_trainable = 0
        for name, module in model.named_modules():
            if isinstance(module, SPECTRAMoE):
                if hasattr(module, 'experts') and len(module.experts) > 0:
                    for expert in module.experts:
                        for param in expert.parameters(recurse=True):
                            param.requires_grad = True
                            expert_params_made_trainable += 1
            # Handle Qwen3 VL MoE experts directly
            if hasattr(module, '__class__') and 'Qwen3VLMoeTextExperts' in module.__class__.__name__:
                for param in module.parameters(recurse=True):
                    param.requires_grad = True
                    expert_params_made_trainable += 1
        logger.info(f"✅ Set {expert_params_made_trainable} expert parameters as trainable")

    # Final model stability fixes
    ensure_router_parameters_trainable(model, logger)
    if model_config.get("freeze_vision"):
        protect_vision_freeze(model, logger)

    # Training Arguments - match train_spectra.py exactly
    training_args = SFTConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
        bf16=training_config.get("bf16", True),
        max_length=data_config["max_seq_length"],
        deepspeed=model_config.get("deepspeed_config"),
        report_to="wandb" if "wandb" in training_config.get("report_to", []) else "none",
        run_name=training_config.get("run_name", "spectra_light"),
        logging_steps=1,
        save_strategy="no",
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        optim=training_config.get("optim", "adamw_bnb_8bit"),
    )

    # Initialize Trainer (Use SymmetricSFTTrainer for ZeRO-3 + VLM stability)
    trainer = SymmetricSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer_or_processor,
        data_collator=collate_fn,
    )

    # Final cleanup and train
    clear_gpu_memory(logger)
    logger.info("🚀 Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
