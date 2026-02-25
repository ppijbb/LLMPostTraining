"""
Memory management utilities for training
"""
import gc
import logging
from typing import Optional
import torch


def clear_gpu_memory(logger: Optional[logging.Logger] = None):
    """Clear GPU memory and run garbage collection with detailed logging"""
    if logger:
        logger.info("🧹 Starting GPU memory cleanup...")
    
    # Log memory before cleanup
    memory_before = None
    if logger:
        from core.training_utils.logging_utils import log_gpu_memory
        memory_before = log_gpu_memory(logger, "BEFORE_CLEANUP")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if logger:
            logger.debug("🧹 CUDA cache cleared and synchronized")
    
    # Force garbage collection
    collected = gc.collect()
    if logger:
        logger.debug(f"🧹 Garbage collection freed {collected} objects")
    
    # Log memory after cleanup
    memory_after = None
    if logger:
        memory_after = log_gpu_memory(logger, "AFTER_CLEANUP")
    
    if logger and memory_before and memory_after:
        freed_allocated = memory_before['allocated'] - memory_after['allocated']
        freed_reserved = memory_before['reserved'] - memory_after['reserved']
        logger.info(f"🧹 Memory cleanup completed - Freed: {freed_allocated:.2f}GB allocated, {freed_reserved:.2f}GB reserved")
    elif logger:
        logger.info("🧹 Memory cleanup completed")


def eval_with_memory_optimization(trainer, original_eval_fn, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", logger: Optional[logging.Logger] = None):
    """Memory-optimized evaluation"""
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    clear_gpu_memory(logger)
    trainer.model.eval()
    original_gc = trainer.args.gradient_checkpointing
    trainer.args.gradient_checkpointing = False
    try:
        with torch.no_grad():
            return original_eval_fn(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    finally:
        trainer.model.train()
        trainer.args.gradient_checkpointing = original_gc
        clear_gpu_memory(logger)

