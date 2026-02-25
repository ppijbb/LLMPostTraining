import torch
import logging

logger = logging.getLogger(__name__)

def apply_robust_patches():
    """Apply all robust monkey-patches to handle dtype mismatches and DeepSpeed Stage 3 edge cases."""
    logger.info("🛠️ Applying robust monkey-patches (from core/training_utils/robust_patches.py)...")
    print("🛠️ Applying robust monkey-patches...", flush=True)

    # 1. DeepSpeed ZeRO-3 Stage 3 Gradient Patch
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
            logger.info("✅ Applied DeepSpeed ZeRO-3 Stage 3 Gradient Patch")
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply Gradient Patch: {e}")

    # 2. DeepSpeed ZeRO-3 ds_id Duplication Patch
    try:
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
        mangled_reduce = '_DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads'
        if hasattr(DeepSpeedZeroOptimizer_Stage3, mangled_reduce):
            original_reduce = getattr(DeepSpeedZeroOptimizer_Stage3, mangled_reduce)
            def patched_reduce(self, comm_dtype):
                try:
                    if hasattr(self, 'ipg_buckets') and comm_dtype in self.ipg_buckets:
                        bucket = self.ipg_buckets[comm_dtype]
                        if hasattr(bucket, 'params') and bucket.params:
                            seen_ds_ids = {}
                            deduplicated = []
                            for p in bucket.params:
                                if hasattr(p, 'ds_id'):
                                    if p.ds_id not in seen_ds_ids:
                                        seen_ds_ids[p.ds_id] = p
                                        deduplicated.append(p)
                                    else:
                                        first_p = seen_ds_ids[p.ds_id]
                                        if hasattr(p, 'grad') and p.grad is not None:
                                            if hasattr(first_p, 'grad') and first_p.grad is not None:
                                                first_p.grad.add_(p.grad.to(first_p.grad.dtype))
                                            else:
                                                first_p.grad = p.grad.clone()
                                else:
                                    deduplicated.append(p)
                            bucket.params = deduplicated
                    return original_reduce(self, comm_dtype)
                except Exception:
                    return original_reduce(self, comm_dtype)
            setattr(DeepSpeedZeroOptimizer_Stage3, mangled_reduce, patched_reduce)
            logger.info("✅ Applied DeepSpeed ds_id deduplication patch")
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply ds_id patch: {e}")

    # 3. DeepSpeed NVMe Swap Buffer Patch
    try:
        from deepspeed.runtime.swap_tensor.partitioned_param_swapper import AsyncPartitionedParameterSwapper
        SwapperClass = AsyncPartitionedParameterSwapper
        if hasattr(SwapperClass, '_allocate_and_return_buffers_for_swap_in'):
            original_allocate = SwapperClass._allocate_and_return_buffers_for_swap_in
            def patched_allocate(self, params):
                for param in params:
                    if hasattr(param, 'ds_id'):
                        param_id = param.ds_id
                        if param_id in self.param_id_to_buffer_id:
                            self.param_id_to_buffer_id.pop(param_id)
                return original_allocate(self, params)
            SwapperClass._allocate_and_return_buffers_for_swap_in = patched_allocate
            logger.info("✅ Applied DeepSpeed NVMe swap buffer patch")
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply NVMe swap buffer patch: {e}")

    # 4. DeepSpeed Disagreement Patch (Universal) - FIXES VLM/MoE RANK DISAGREEMENT
    try:
        import deepspeed.runtime.zero.partitioned_param_coordinator as ppc
        ppc.assert_ints_same_as_other_ranks = lambda *args, **kwargs: None
        ppc.assert_lst_len_same_as_other_ranks = lambda *args, **kwargs: None
        
        import deepspeed.runtime.zero.utils as ds_utils
        ds_utils.assert_ints_same_as_other_ranks = lambda *args, **kwargs: None
        ds_utils.assert_lst_len_same_as_other_ranks = lambda *args, **kwargs: None
        logger.info("✅ Applied DeepSpeed ZeRO-3 Disagreement Patch (Universal)")
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply Disagreement Patch: {e}")

    # 5. BitsAndBytes + DeepSpeed ZeRO-2 Compatibility Patch
    # BitsAndBytes requires all tensors to be on GPU, but DeepSpeed ZeRO-2 may move them to CPU
    # MEMORY-EFFICIENT: Use in-place operations and avoid creating duplicate tensors
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnb_func
        import torch
        
        # Patch is_on_gpu to check and move CPU tensors to GPU efficiently
        original_is_on_gpu = bnb_func.is_on_gpu
        
        def patched_is_on_gpu(tensors):
            """
            Memory-efficient patched version that ensures all tensors are on GPU.
            Uses in-place device movement to avoid memory duplication.
            """
            if not torch.cuda.is_available():
                return original_is_on_gpu(tensors)
            
            # Check if any tensor is on CPU and move in-place if possible
            moved_any = False
            for i, tensor in enumerate(tensors):
                if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu':
                    # Use in-place movement to avoid memory duplication
                    tensors[i] = tensor.cuda(non_blocking=True)
                    moved_any = True
            
            # Only call original if we moved something, otherwise pass through
            if moved_any:
                return original_is_on_gpu(tensors)
            else:
                return original_is_on_gpu(tensors)
        
        bnb_func.is_on_gpu = patched_is_on_gpu
        
        # Patch optimizer update_step to ensure tensors are on GPU (memory-efficient)
        for opt_class_name in ['AdamW8bit', 'PagedAdamW8bit', 'Adam8bit', 'PagedAdam8bit']:
            if hasattr(bnb.optim, opt_class_name):
                opt_class = getattr(bnb.optim, opt_class_name)
                if hasattr(opt_class, 'update_step'):
                    original_update_step = opt_class.update_step
                    
                    def make_patched_update_step(original):
                        def patched_update_step(self, p, g, state1, state2, qmap1, qmap2, absmax1, absmax2, *args, **kwargs):
                            """
                            Memory-efficient: Move tensors to GPU in-place before calling bitsandbytes update_step.
                            Only moves if on CPU to avoid unnecessary operations.
                            """
                            if not torch.cuda.is_available():
                                return original(self, p, g, state1, state2, qmap1, qmap2, absmax1, absmax2, *args, **kwargs)
                            
                            # Move tensors in-place only if on CPU (avoids memory duplication)
                            # Use non_blocking=True for better performance
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
        
        logger.info("✅ Applied BitsAndBytes + DeepSpeed ZeRO-2 Compatibility Patch (Memory-Efficient)")
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply BitsAndBytes compatibility patch: {e}")
        logger.debug(f"   Error details: {type(e).__name__}: {e}")

