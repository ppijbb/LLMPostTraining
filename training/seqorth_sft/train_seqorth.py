#!/usr/bin/env python3
"""
Seqorth SFT Training Script using Config File
"""

import os
import sys
import json
import warnings
import contextlib
import types

# --- Suppress Annoying Deprecation Warnings ---
# These are caused by library drift (e.g. cryptography, scipy) and flood the logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except ImportError:
    pass
# ----------------------------------------------
# [0vs2048] Global Participation Pool for v16 GAPH
_PARTICIPATION_POOL = []

# --- JIT cpp_extension: avoid multi-rank compile deadlock (must be before any torch import) ---
# Each rank compiles in its own dir so no lock contention; unset TORCH_CUDA_ARCH_LIST triggers "all archs" + slow compile.
if "TORCH_EXTENSION_DIR" not in os.environ:
    _rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    os.environ["TORCH_EXTENSION_DIR"] = f"/tmp/torch_ext_rank{_rank}"
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("SEQORTH_CUDA_ARCH", "8.0")

# --- Global Monkey Patches for ZeRO-3 Compatibility ---
import torch.nn as nn

def _patch_nn_init():
    import torch.nn.init as torch_init
    
    orig_xavier_uniform = torch_init.xavier_uniform_
    def safe_xavier_uniform(tensor, gain=1.0):
        if tensor.is_meta: return tensor
        if tensor.numel() == 0: return tensor
        if tensor.ndimension() < 2:
            return torch_init.uniform_(tensor, -0.02, 0.02)
        return orig_xavier_uniform(tensor, gain)
    torch_init.xavier_uniform_ = safe_xavier_uniform
    nn.init.xavier_uniform_ = safe_xavier_uniform

    orig_xavier_normal = torch_init.xavier_normal_
    def safe_xavier_normal(tensor, gain=1.0):
        if tensor.is_meta: return tensor
        if tensor.numel() == 0: return tensor
        if tensor.ndimension() < 2:
            return torch_init.normal_(tensor, std=0.02)
        return orig_xavier_normal(tensor, gain)
    torch_init.xavier_normal_ = safe_xavier_normal
    nn.init.xavier_normal_ = safe_xavier_normal

    orig_kaiming_uniform = torch_init.kaiming_uniform_
    def safe_kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        if tensor.is_meta: return tensor
        if tensor.numel() == 0: return tensor
        if tensor.ndimension() < 2:
            return torch_init.uniform_(tensor, -0.02, 0.02)
        return orig_kaiming_uniform(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    torch_init.kaiming_uniform_ = safe_kaiming_uniform
    nn.init.kaiming_uniform_ = safe_kaiming_uniform

    orig_orthogonal = torch_init.orthogonal_
    def safe_orthogonal(tensor, gain=1.0):
        if tensor.is_meta: return tensor
        if tensor.numel() == 0: return tensor
        if tensor.ndimension() < 2:
            return torch_init.normal_(tensor, std=0.02)
        return orig_orthogonal(tensor, gain)
    torch_init.orthogonal_ = safe_orthogonal
    nn.init.orthogonal_ = safe_orthogonal
    
    # Patch the internal fan calculator too
    if hasattr(torch_init, '_calculate_fan_in_and_fan_out'):
        orig_calc = torch_init._calculate_fan_in_and_fan_out
        def safe_calc(tensor):
            if tensor.numel() == 0: return 1, 1
            if tensor.dim() < 2: return 1, 1
            f_in, f_out = orig_calc(tensor)
            return max(1, f_in), max(1, f_out)
        torch_init._calculate_fan_in_and_fan_out = safe_calc

# --- Patch DeepSpeed ZeRO-3: handle None grad in unscale_and_clip_grads ---
# MoE sparse activation means unused experts have no gradient; their fp32 partition grad is None.
# Mathematically correct: unused param -> gradient = 0.
def _patch_deepspeed_none_grad():
    try:
        import torch as _t
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
        _orig_unscale = DeepSpeedZeroOptimizer_Stage3.unscale_and_clip_grads

        def _safe_unscale(self, sub_group_id, scaled_global_grad_norm):
            flat = self.fp32_partitioned_groups_flat[sub_group_id]
            if flat.grad is None:
                flat.grad = _t.zeros_like(flat)
            return _orig_unscale(self, sub_group_id, scaled_global_grad_norm)

        DeepSpeedZeroOptimizer_Stage3.unscale_and_clip_grads = _safe_unscale
    except Exception:
        pass

_patch_deepspeed_none_grad()

def _patch_qwen3_participation_guard():
    """
    ULTIMATE ZeRO-2/3 & AutoTP Fix (v10): Root Participation Guard (RPG).
    1) RPG: Adds all parameters to the autograd graph via the loss to prevent any '0 vs 2048' mismatch.
    2) MV1: Ensures Experts and Vision components process at least size 1 volume.
    """
    try:
        import torch
        import torch.nn.functional as F
        import torch.distributed as dist
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextExperts, 
            Qwen3VLMoeTextRMSNorm,
            Qwen3VLMoeVisionPatchMerger,
            Qwen3VLMoeVisionBlock,
            Qwen3VLMoeTextMLP,
            Qwen3VLMoeTextAttention
        )
        
        # [0vs2048] 커스텀 autograd Function: Linear backward를 직접 제어하여 grad_output (0,2048) 방지
        # VisionPatchMerger와 VisionMLP 모두에서 사용
        def _vision_rank():
            try:
                return dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                return 0

        # [0vs2048] Participation Pool (v17)
        # We collect all dummy forward results and add them to the loss at the very end.
        def _record_participation(x):
            global _PARTICIPATION_POOL
            if x is None: return
            if torch.is_tensor(x):
                if x.requires_grad:
                    _PARTICIPATION_POOL.append(x.sum() * 0.0)
            elif isinstance(x, (list, tuple)):
                for o in x: _record_participation(o)
            elif isinstance(x, dict):
                for v in x.values(): _record_participation(v)
        
        # [0vs2048] SafeLinearFunction: backward에서 grad_output/grad_weight shape 보정 (0 vs 2048/4608 방지)
        # ZeRO-2에서도 vision path backward 시 AccumulateGrad shape mismatch 발생 → 커스텀 backward로 해결
        class SafeLinearFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, weight, bias, seq_len):
                ctx.save_for_backward(input, weight, bias)
                ctx.seq_len = seq_len
                return F.linear(input, weight, bias)

            @staticmethod
            def backward(ctx, grad_output):
                input, weight, bias = ctx.saved_tensors
                seq_len = ctx.seq_len
                r2 = _vision_rank()
                if grad_output is not None and grad_output.dim() >= 1:
                    if grad_output.shape[0] == 0:
                        grad_output = torch.zeros(seq_len, *grad_output.shape[1:], device=grad_output.device, dtype=grad_output.dtype)
                    elif grad_output.dim() >= 2 and (grad_output.shape[1] == 0 or grad_output.shape[1] != weight.shape[0]):
                        grad_output = torch.zeros(grad_output.shape[0], weight.shape[0], device=grad_output.device, dtype=grad_output.dtype)
                grad_input = grad_output @ weight if grad_output is not None else None
                if grad_output is not None and input is not None:
                    try:
                        grad_weight = grad_output.t() @ input
                        if grad_weight.shape != weight.shape:
                            grad_weight = torch.zeros_like(weight)
                    except RuntimeError:
                        grad_weight = torch.zeros_like(weight)
                else:
                    grad_weight = None
                grad_bias = grad_output.sum(0) if grad_output is not None and bias is not None else None
                return grad_input, grad_weight, grad_bias, None

        # --- 2) Vision: MV1 Guards ---
        # Vision & Text components: Participation Guard (v16 - Global Connection)
        def mv1_module_forward(orig_fn):
            def wrapped(self, *args, **kwargs):
                if not self.training: 
                    return orig_fn(self, *args, **kwargs)
                
                # [0vs2048] Extract focal input 'x' (hidden_states)
                x = kwargs.get('hidden_states', kwargs.get('x', args[0] if args else None))
                
                has_data = False
                if torch.is_tensor(x) and x.numel() > 0:
                    has_data = True
                elif isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]) and x[0].numel() > 0:
                    has_data = True
                
                if has_data:
                    return orig_fn(self, *args, **kwargs)
                
                # Participatory Dummy path
                # Use x's properties if available, else defaults
                device = x.device if torch.is_tensor(x) else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                dtype = x.dtype if torch.is_tensor(x) else torch.bfloat16
                h_size = x.shape[-1] if (torch.is_tensor(x) and x.dim() > 0) else getattr(self, "hidden_size", 4096)
                
                dummy_x = torch.zeros((1, h_size), device=device, dtype=dtype, requires_grad=True)
                
                # Construct dummy call inputs
                new_args = list(args)
                new_kwargs = kwargs.copy()
                if args:
                    new_args[0] = dummy_x
                elif 'hidden_states' in new_kwargs:
                    new_kwargs['hidden_states'] = dummy_x
                elif 'x' in new_kwargs:
                    new_kwargs['x'] = dummy_x
                else:
                    new_args = [dummy_x] # Fallback
                
                # Special handling for Attention (cos/sin lengths)
                if "position_embeddings" in new_kwargs:
                    pe = new_kwargs["position_embeddings"]
                    if isinstance(pe, (list, tuple)) and len(pe) >= 2:
                        new_kwargs["position_embeddings"] = tuple(t[:, :1] if torch.is_tensor(t) else t for t in pe)
                
                # Special handling for Vision Blocks (cu_seqlens)
                if "cu_seqlens" in new_kwargs:
                    new_kwargs["cu_seqlens"] = torch.tensor([0, 1], device=device, dtype=torch.int32)
                
                out = orig_fn(self, *new_args, **new_kwargs)
                
                # Connection logic (v17): record in pool and return empty
                _record_participation(out)
                
                if isinstance(out, (list, tuple)):
                    return tuple(o[:0] if torch.is_tensor(o) else o for o in out)
                return out[:0]
            return wrapped

        # VisionPatchMerger: 빈 입력 시 원본 forward 호출 금지 → F.linear backward에서 0 vs 2048 방지.
        # dummy forward 없이 출력 shape만 맞춰 zeros 반환 (그래프 미연결).
        # 0-size 원인: Seqorth 아님. baseline에서도 Qwen3-VL vision path에 빈 배치(이미지 0개/0 patches)가
        # 들어가면 F.linear backward에서 gradient (0,...) vs param (2048,...) → AccumulateGrad 에러.
        _merger_orig = Qwen3VLMoeVisionPatchMerger.forward
        def vision_merger_empty_guard(self, x: torch.Tensor) -> torch.Tensor:
            if not self.training:
                return _merger_orig(self, x)
            
            has_data = torch.is_tensor(x) and x.numel() > 0 and x.shape[0] > 0
            if has_data:
                # [0vs2048] 데이터 있을 때 SafeLinearFunction 사용 → backward에서 0 vs 2048 방지
                seq = x.shape[0]
                x_norm = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
                x_fc1 = SafeLinearFunction.apply(x_norm, self.linear_fc1.weight, self.linear_fc1.bias, seq)
                x_act = self.act_fn(x_fc1)
                out = SafeLinearFunction.apply(x_act, self.linear_fc2.weight, self.linear_fc2.bias, seq)
                return out
            
            # [0vs2048] 빈 입력: Participatory Dummy
            r = _vision_rank()
            print(f"[0vs2048] rank={r} VisionPatchMerger BYPASS-WITH-DUMMY (empty input)", flush=True)
            dummy_x = torch.zeros((1, self.hidden_size), device=x.device, dtype=x.dtype, requires_grad=True)
            out = _merger_orig(self, dummy_x)
            _record_participation(out)
            return out[:0]
        Qwen3VLMoeVisionPatchMerger.forward = vision_merger_empty_guard

        # VisionBlock: 빈 hidden_states 시 Linear/Attn backward에서 0 vs 2048 발생 → bypass.
        _block_orig = Qwen3VLMoeVisionBlock.forward
        def vision_block_empty_guard(self, hidden_states, cu_seqlens=None, rotary_pos_emb=None, position_embeddings=None, **kwargs):
            if not self.training:
                return _block_orig(self, hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings, **kwargs)
            
            has_data = torch.is_tensor(hidden_states) and hidden_states.numel() > 0
            if has_data:
                return _block_orig(self, hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings, **kwargs)
            
            r = _vision_rank()
            print(f"[0vs2048] rank={r} VisionBlock BYPASS-WITH-DUMMY (empty input)", flush=True)
            
            # Dummy Path (v16): Use 1 token
            h_size = self.norm1.weight.shape[0]
            dummy_hs = torch.zeros((1, h_size), device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True)
            dummy_cu = torch.tensor([0, 1], device=hidden_states.device, dtype=torch.int32)
            
            out = _block_orig(self, dummy_hs, cu_seqlens=dummy_cu, rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings, **kwargs)
            _record_participation(out)
            return out[:0]
        Qwen3VLMoeVisionBlock.forward = vision_block_empty_guard

        # [0vs2048] VisionMLP: linear_fc1(1152→4608), linear_fc2(4608→1152) backward에서 0 vs 4608 방지
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeVisionMLP
        _mlp_orig = Qwen3VLMoeVisionMLP.forward
        def vision_mlp_safe_guard(self, hidden_state):
            if not self.training:
                return _mlp_orig(self, hidden_state)
            seq = hidden_state.shape[0]
            x_fc1 = SafeLinearFunction.apply(hidden_state, self.linear_fc1.weight, self.linear_fc1.bias, seq)
            x_act = self.act_fn(x_fc1)
            out = SafeLinearFunction.apply(x_act, self.linear_fc2.weight, self.linear_fc2.bias, seq)
            return out
        Qwen3VLMoeVisionMLP.forward = vision_mlp_safe_guard

        # [0vs2048] Removed complex mv1_module_forward wrappers (v19)
        # We now rely on Input Injection (in compute_loss) to guarantee Vision Tower participation.
        # This is safer and less intrusive than wrapping every module.

        print("✅ Qwen3 Input Injection Guard (v19) enabled", flush=True)

        # --- 2) Experts: Synchronized MV1 Loop ---
        _exp_orig_fwd = Qwen3VLMoeTextExperts.forward
        def patched_exp_forward(self, hidden_states, routing_weights, router_indices):
            if not self.training:
                return _exp_orig_fwd(self, hidden_states, routing_weights, router_indices)
            
            batch_size = hidden_states.shape[0]
            hidden_states_flat = hidden_states.reshape(-1, self.hidden_size)
            next_states = torch.zeros_like(hidden_states_flat)
            device = hidden_states.device
            dtype = hidden_states.dtype

            with torch.no_grad():
                local_mask = torch.zeros(self.num_experts, device=device, dtype=torch.float32)
                if router_indices.numel() > 0:
                    unique_indices = router_indices.unique()
                    local_mask.scatter_(0, unique_indices, 1.0)
                if dist.is_initialized():
                    dist.all_reduce(local_mask, op=dist.ReduceOp.MAX)
                global_expert_hit = local_mask.nonzero().view(-1)

            for expert_idx_scalar in global_expert_hit:
                expert_idx = expert_idx_scalar.item()
                token_mask = (router_indices == expert_idx).any(dim=-1)
                token_idx = torch.where(token_mask)[0]

                if token_idx.numel() > 0:
                    current_state = hidden_states_flat[token_idx]
                    is_dummy = False
                else:
                    # Participation Guard (v16): requires_grad=True is life
                    current_state = torch.zeros((1, self.hidden_size), device=device, dtype=dtype, requires_grad=True)
                    is_dummy = True

                w_gate_up = self.gate_up_proj[expert_idx]
                w_down = self.down_proj[expert_idx]
                gate_up = F.linear(current_state, w_gate_up.T)
                gate, up = gate_up.chunk(2, dim=-1)
                out = F.linear(up * self.act_fn(gate), w_down.T)
                
                if not is_dummy:
                    w = routing_weights[token_idx, expert_idx].unsqueeze(-1)
                    next_states.index_add_(0, token_idx, (out * w).to(next_states.dtype))
                else:
                    # v16: Record dummy expert path
                    _record_participation(out)

            return next_states.view(batch_size, -1, self.hidden_size)
        
        Qwen3VLMoeTextExperts.forward = patched_exp_forward
        print("✅ Qwen3 Participation Guard (v16 - Global Pool) applied", flush=True)

    except Exception as e:
        import traceback
        print(f"⚠️ Qwen3 participation guard patch failed: {e}", flush=True)
        traceback.print_exc()

def _patch_trainer_for_participation_guard(trainer):
    """
    Guaranteed Participation Guard (v16):
    Combines module-level pool terms and root-level parameter sums to ensure
    100% autograd participation across all sharded and unsharded parameters.
    """
    if not hasattr(trainer, 'compute_loss'): return
    
    _orig_compute_loss = trainer.compute_loss
    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Clear pool before forward
        # NOTE: Using a simple list lookup since it's defined in the same scope in train_seqorth.py
        try:
            from types import SimpleNamespace
            global _PARTICIPATION_POOL
        except ImportError: pass
        
        if '_PARTICIPATION_POOL' in globals():
            globals()['_PARTICIPATION_POOL'].clear()
            
        try:
            res = _orig_compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            res = _orig_compute_loss(model, inputs, return_outputs=return_outputs)
            
        loss = res[0] if return_outputs else res
        
        # ROOT GUARD: Ensure every parameter shard participates (ZeRO-3 safe: skip empty partitions to avoid "got [0] vs expected [N,M]" SumBackward0)
        guard = 0.0
        for p in model.parameters():
            if p.requires_grad and p.numel() > 0:
                guard = guard + (p.sum() * 0.0)
        
        # POOL GUARD: Ensure every skipped module (dummy forward) participates
        if '_PARTICIPATION_POOL' in globals():
            pool = globals()['_PARTICIPATION_POOL']
            while pool:
                guard = guard + pool.pop()
        
        # Final connection
        if return_outputs:
            return (loss + guard, res[1])
        else:
            return loss + guard

    trainer.compute_loss = types.MethodType(patched_compute_loss, trainer)
    print("✅ Trainer Participation Guard (v16 - GAPH) applied", flush=True)

def _get_trainable_parameters(model):
    """PEFTWrapper 대응 파라미터 수집"""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model.parameters()
    return model.parameters()
_patch_universal_participation = _patch_qwen3_participation_guard
_patch_universal_participation()


# [ZeRO-3 Fix] Patch transformers to allow loading weights into partitioned (size 0) parameters
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
    # 만약 mismatched_keys가 이미 비어있다면 (ignore_mismatched_sizes=False인 경우 등) 바로 반환
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
    
    # ZeRO-3 sharding으로 인해 size가 0이 된 파라미터들을 오탐지 목록에서 제거
    filtered_keys = []
    filtered_shapes = []
    
    # 모델의 실제 파라미터 맵 (ds_id 확인용)
    named_params = dict(model.named_parameters())
    
    for key, (shape1, shape2) in zip(mismatched_keys, mismatched_shapes):
        # shape2가 model의 shape. 만약 이게 0이거나 비어있다면 ZeRO-3 placeholder일 가능성이 높음
        is_zero_size = (len(shape2) == 0 or (len(shape2) == 1 and shape2[0] == 0))
        
        if is_zero_size:
            param = named_params.get(key)
            if param is not None and hasattr(param, "ds_id"):
                # DeepSpeed가 관리하는 파라미터이므로 mismatch가 아님 (성공적으로 shard됨)
                continue
                
        filtered_keys.append(key)
        filtered_shapes.append((shape1, shape2))
        
    return filtered_keys, filtered_shapes

# 패치 적용
modeling_utils._find_mismatched_keys = _safe_find_mismatched_keys
# ---------------------------------------------------

import torch
import torch.distributed as dist
import gc
# CRITICAL: Set device IMMEDIATELY to prevent all ranks defaulting to GPU 0
# This must happen before any other library imports processing!
# Helper function to check if this is main process
def is_main_process():
    """Check if this is the main process (rank 0)"""
    try:
        # Use global dist import (already imported at top of file)
        if dist.is_initialized():
            return dist.get_rank() == 0
        else:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
            return rank == 0
    except:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        return rank == 0

local_rank_env = os.environ.get("LOCAL_RANK")
if local_rank_env is not None:
    try:
        local_rank = int(local_rank_env)
        if -1 < local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
            if is_main_process():
                print(f"✅ [Pre-Init] Set CUDA device to {local_rank}")
                print(f"   🔍 Verifying: torch.cuda.current_device() == {torch.cuda.current_device()}")
    except ValueError:
        pass
import traceback
import argparse
import logging
import time
import types
import inspect
import warnings

from datetime import datetime
from typing import Dict, Any
from torchinfo import summary
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available, is_flash_attn_3_available
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq
)
import sys
from transformers import logging as transformers_logging

from transformers.trainer_utils import set_seed
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import TaskType
try:
    from peft.utils.other import ModulesToSaveWrapper
except ImportError:
    try:
        from peft.utils import ModulesToSaveWrapper
    except ImportError:
        try:
            from peft.tuners.tuners_utils import ModulesToSaveWrapper
        except ImportError:
            ModulesToSaveWrapper = None
import wandb
import accelerate.utils.dataclasses as acc_dataclasses
import deepspeed  # Required for deepspeed.zero.Init
try:
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
except ImportError:
    ZeroParamStatus = None
# Expect PYTHONPATH to include repo root (e.g. run from repo root or set PYTHONPATH)

# Import custom modules  
from models import (
    SeqorthForCausalLM, 
    SeqorthConfig, 
    SeqorthForConditionalGeneration, 
    SeqorthTextConfig, 
    SeqorthTextModel, 
    SeqorthModel,
    SeqorthExoskeletonMoEInjector # Universal Exoskeleton
)

from core.training_utils import (
    format_parameters, 
    load_config, 
    setup_deepspeed_environment,
    setup_logging,
    log_gpu_memory,
    log_error_context,
    save_oom_error_info,
    handle_cuda_oom,
    handle_ram_oom,
    handle_training_exception,
    collect_environment_info,
    clear_gpu_memory,
    eval_with_memory_optimization,
    count_unique_parameters,
    check_model_size,
    get_dynamic_lora_target_modules,
    ensure_router_parameters_trainable,
    ensure_router_in_optimizer,
    verify_model_consistency,
    protect_vision_freeze,

    setup_dataset,
    create_training_args,
    BatchTrackingCallback,
    DetailedTrainingCallback,
    ModulesToSaveSyncCallback
)

from evaluation.moe_monitoring_callback import TorchMoECallback
from evaluation.callbacks import ModelEvalCallback
# IFEval is now integrated into ModelEvalCallback - no separate callback needed
# from evaluation.ifeval_callback import IFEvalCallback
from evaluation.moe_monitoring_callback import create_moe_callback_for_transformers
from evaluation.router_weight_callback import RouterWeightTrackingCallback

# --- Additional Global Imports (moved from local imports) ---
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import AsyncPartitionedParameterSwapper
import deepspeed.runtime.zero.partitioned_param_coordinator as ppc
import deepspeed.runtime.zero.utils as ds_utils
from accelerate import Accelerator
from deepspeed.runtime.config import DeepSpeedConfig
import yaml
from pathlib import Path
import deepspeed.runtime.zero.partition_parameters as pp_module
import deepspeed.runtime.engine
import shutil
import deepspeed.runtime.zero.linear as ds_zero_linear
from deepspeed.runtime.zero.linear import autocast_custom_fwd
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers.modeling_utils import VLMS
from transformers.models.siglip import SiglipVisionConfig
import tempfile
from transformers import AutoModel
from transformers.integrations import HfDeepSpeedConfig
import torch.nn as nn
from models.seqorth_model import SeqorthPreTrainedModel
from models.seqorth_model import SeqorthRouter
from models.seqorth_model import SeqorthMoE
from transformers import AutoConfig
from deepspeed.runtime.zenflow.zenflow_stage_1_and_2 import ZenFlowZeroOptimizer
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

import deepspeed.utils
from torch.profiler import profile, record_function, ProfilerActivity

# Standard DeepSpeed patches are disabled to rely on standard Trainer/Accelerate integration

def apply_universal_exoskeleton_guard(logger):
    """
    ULTIMATE ZeRO-3 Fix: Radically identifies and protects EVERY weight-carrying sub-module
    from being split by ZeRO-3. This is the only way to prevent 'Size 0' loading errors
    across any transformer architecture.
    """
    import sys
    import torch.nn as nn
    
    # 1. Expanded patterns for atomic units (sharding leaves)
    # We add common layer names and our custom Seqorth classes
    atomic_patterns = [
        "Block", "Layer", "Merger", "MLP", "Attention", "Expert", "Router", 
        "Projector", "Vision", "Encoder", "Embed", "Rotary", "Patch", "Tower", 
        "Bridge", "Head", "Decoder", "Transformer", "PatchEmbed", "Positional",
        "SeqorthMoE", "SeqorthRouter" # Safeguard our own additions
    ]
    
    # 2. Deep Scan every single module for potential leaf classes
    # We look for ANY class that is an nn.Module and matches our patterns
    potential_leaves = set()
    for mod_name, module in list(sys.modules.items()):
        # Scan almost everything to be safe with remote code
        for attr_name in dir(module):
            try:
                cls = getattr(module, attr_name)
                if isinstance(cls, type) and issubclass(cls, nn.Module):
                    if any(p in attr_name for p in atomic_patterns):
                        # Skip generic small wrappers
                        if attr_name not in ["ModuleList", "Sequential", "ModuleDict"]:
                            potential_leaves.add(attr_name)
            except:
                continue
    
    # 3. Aggressively inject leaves into ANY class that could be a sharding container
    # This includes Model, Config, and nested structural classes
    container_patterns = ["Model", "Config", "Decoder", "Encoder", "Tower", "Generation"]
    patched_count = 0
    
    for mod_name, module in list(sys.modules.items()):
        for attr_name in dir(module):
            try:
                cls = getattr(module, attr_name)
                if not isinstance(cls, type):
                    continue
                
                # If it looks like a container or already has sharding info
                is_container = (
                    any(p in attr_name for p in container_patterns) or 
                    hasattr(cls, "_no_split_modules")
                )
                
                if is_container:
                    if not hasattr(cls, "_no_split_modules") or cls._no_split_modules is None:
                        cls._no_split_modules = []
                    
                    # Convert to set for fast check and back to list
                    current_leaves = set(cls._no_split_modules)
                    added = []
                    for leaf in potential_leaves:
                        # Don't add a class to its own no-split list
                        if leaf != attr_name and leaf not in current_leaves:
                            cls._no_split_modules.append(leaf)
                            added.append(leaf)
                    
                    if added:
                        patched_count += 1
                        # Log only for significant classes to avoid spam
            except:
                continue
    
    return patched_count > 0

def apply_deepspeed_shutil_patch():
    try:
        import deepspeed.runtime.engine as ds_engine
        if hasattr(ds_engine, 'shutil') and hasattr(ds_engine.shutil, 'copytree'):
            original_copytree = ds_engine.shutil.copytree
            def patched_copytree(src, dst, **kwargs):
                kwargs['dirs_exist_ok'] = True
                return original_copytree(src, dst, **kwargs)
            ds_engine.shutil.copytree = patched_copytree
            print("✅ Patched deepspeed.runtime.engine.shutil.copytree with dirs_exist_ok=True")
    except Exception as e:
        print(f"⚠️ Failed to patch deepspeed shutil: {e}")

apply_deepspeed_shutil_patch()

def apply_deepspeed_zero3_norm_patch():
    """
    ZeRO-3 OOM fix:
    DeepSpeed stage3 default _constant_buffered_norm2 uses part.double(), which can
    trigger a large temporary fp64 allocation during backward on large grad buffers.
    Patch it to fp32 + smaller chunks to reduce peak memory.
    """
    try:
        import deepspeed.runtime.zero.stage3 as ds_stage3
        cls = ds_stage3.DeepSpeedZeroOptimizer_Stage3
        if getattr(cls, "_seqorth_norm_patch_applied", False):
            return

        def _seqorth_constant_buffered_norm2(self, input, buffer_size=8_388_608):
            if input is None:
                return torch.tensor(0.0, device="cpu")

            flat = input.view(-1)
            if flat.numel() == 0:
                return torch.tensor(0.0, device=flat.device, dtype=torch.float32)

            norm_sq = None
            for part in flat.split(buffer_size):
                # fp32 accumulation is enough for grad norm and avoids fp64 OOM spikes.
                part_norm_sq = part.float().norm(2).pow(2)
                norm_sq = part_norm_sq if norm_sq is None else norm_sq + part_norm_sq
            return norm_sq.sqrt()

        cls._constant_buffered_norm2 = _seqorth_constant_buffered_norm2
        cls._seqorth_norm_patch_applied = True
        print("✅ Patched DeepSpeed ZeRO-3 grad norm to fp32 chunked path")
    except Exception as e:
        print(f"⚠️ Failed to patch DeepSpeed ZeRO-3 grad norm: {e}")

apply_deepspeed_zero3_norm_patch()

# --- Standard Setup ---
# Standard DeepSpeed patches are disabled to rely on standard Trainer/Accelerate integration

try:
    # AutoConfig.register("seqorth", SeqorthConfig)
    AutoConfig.register("seqorth", SeqorthConfig)
    AutoConfig.register("seqorth_text", SeqorthTextConfig)
    AutoModel.register(SeqorthConfig, SeqorthModel)
    AutoModel.register(SeqorthTextConfig, SeqorthTextModel)
    AutoModelForCausalLM.register(SeqorthConfig, SeqorthForConditionalGeneration)

    VLMS.append("seqorth")
except Exception as e:
    traceback.format_exc()
    print(f"Failed to register Seqorth model: {e}")
    print("Seqorth cannot train without registering model... exiting...")
    raise e

transformers_logging.enable_progress_bar()
transformers_logging.set_verbosity_warning()
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR) 

# Global logger instance - only main process outputs to console
logger = setup_logging()


def run_post_training_validation(model_path: str, training_config_path: str, output_dir: str, device: str = "cuda"):
    """Post-training validation (간소화)"""
    logger.info("🚀 Post-Training Validation (skipped - use separate validation scripts)")
    return {}


def setup_tokenizer(model_config: Dict[str, Any]):
    """토크나이저만 별도로 로딩 (데이터셋 처리용)"""
    logger.info("🚀 Starting tokenizer setup...")
    
    # Check if initializing from scratch
    initialize_from_scratch = bool(model_config.get("initialize_from_scratch", False))

    # Load tokenizer - 안정적인 로딩 로직
    if initialize_from_scratch:
        # From-scratch initialization MUST specify tokenizer_name_or_path (no hidden defaults)
        if not model_config.get("tokenizer_name_or_path"):
            raise ValueError("initialize_from_scratch=True requires model_config.tokenizer_name_or_path")
        tokenizer_path = model_config["tokenizer_name_or_path"]
        logger.info(f"🔤 Initializing from scratch - loading tokenizer from: {tokenizer_path}")
    else:
        tokenizer_path = model_config.get("tokenizer_name_or_path") or model_config["model_name_or_path"]
        logger.info(f"🔤 Loading tokenizer from: {tokenizer_path}")
    
    tokenizer = None
    try:
        logger.debug("  - Attempting AutoProcessor...")
        tokenizer = AutoProcessor.from_pretrained(
            tokenizer_path,
            trust_remote_code=model_config["trust_remote_code"]
        )
        logger.info("  ✅ AutoProcessor loaded successfully")
    except Exception as e:
        logger.warning(f"  ❌ AutoProcessor failed: {e}")
        try:
            logger.debug("  - Attempting AutoTokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=model_config["trust_remote_code"]
            )
            logger.info("  ✅ AutoTokenizer loaded successfully")
        except Exception as e2:
            logger.error(f"  ❌ AutoTokenizer also failed: {e2}")
            log_error_context(logger, e2, "tokenizer_loading")
            raise RuntimeError(f"토크나이저 로딩 실패: {e2}")
    
    # Set chat template with error handling
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "sft", "chat_template.txt"), "r") as f:
            chat_template = f.read()
        
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.tokenizer.chat_template = chat_template
        else:
            tokenizer.chat_template = chat_template
        logger.debug(f"Chat template set (length: {len(chat_template)})")
    except Exception as e:
        logger.warning(f"Chat template setup failed: {e}")
    
    # Set padding side for multimodal models
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer.tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "right"

    # Ensure tokenizer has pad token
    if not hasattr(tokenizer, 'pad_token'):
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.pad_token = tokenizer.tokenizer.pad_token if tokenizer.tokenizer.pad_token is not None else tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if not hasattr(tokenizer, 'convert_tokens_to_ids'):
        tokenizer.convert_tokens_to_ids = tokenizer.tokenizer.convert_tokens_to_ids
        
    return tokenizer



# ==============================================================================
# CRITICAL: MULTI-LAYER PATCH for _deepstack_process to fix SliceBackward0 error
# ==============================================================================
def apply_seqorth_patch(model, logger):
    """Seqorth 전용 패치: SliceBackward0 오류 수정을 위해 _deepstack_process를 래핑"""
    try:
        def find_target(m):
            if hasattr(m, "_deepstack_process"):
                return m
            for child in m.children():
                res = find_target(child)
                if res: return res
            return None
        
        target = find_target(model)
        if target:
            original_process = target._deepstack_process
            def patched_process(self, hidden_states, *args, **kwargs):
                return original_process(hidden_states.clone(), *args, **kwargs)
            target._deepstack_process = types.MethodType(patched_process, target)
            logger.info(f"✅ Applied SliceBackward0 fix to {type(target).__name__}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply seqorth patch: {e}")


def setup_model(model_config: Dict[str, Any]):
    """모델 설정. (토크나이저는 로드하지 않음) modules_to_save_list를 반환"""
    logger.info("🚀 Starting model setup...")


    # NOTE: Delay DeepSpeed env setup until AFTER model load to avoid HF ZeRO-3 init slow path
    logger.info("🔧 Setting up DeepSpeed environment...")
    setup_deepspeed_environment()
    
    # Check if initializing from scratch
    initialize_from_scratch = bool(model_config.get("initialize_from_scratch", False))
    
    # Decide global model dtype strictly from training_config flags.
    # This must be the single source of truth for bf16/fp16/fp32 to avoid mixed-dtype under ZeRO-3.
    # NOTE: training_config is expected to be provided by the caller (module-global).
    model_dtype = torch.bfloat16 if training_config.get("bf16", False) else (
        torch.float16 if training_config.get("fp16", False) else torch.float32
    )

    def _require_keys(d: Dict[str, Any], keys: list[str], context: str) -> None:
        missing = [k for k in keys if k not in d]
        if missing:
            raise KeyError(f"[{context}] Missing required keys: {missing}")

    # Require explicit attention backend (no silent fallback)
    seqorth_params = model_config.get("seqorth_params", {}).copy() # Use a copy to avoid mutating original
    
    # Fix typo reported by USER: router_entropy_coe -> router_entropy_coef
    if "router_entropy_coe" in seqorth_params:
        seqorth_params["router_entropy_coef"] = seqorth_params.pop("router_entropy_coe")
        logger.info("  🔧 Fixed typo: router_entropy_coe -> router_entropy_coef in seqorth_params")

    # Robust rope_scaling handling
    if "rope_scaling" not in seqorth_params:
        logger.info("  ℹ️  No rope_scaling found in config, using default")
    else:
        # Handle cases where rope_scaling might be partial or malformed
        rs = seqorth_params["rope_scaling"]
        if isinstance(rs, dict):
            if "rope_type" not in rs and "type" in rs:
                rs["rope_type"] = rs["type"]
            if "rope_type" not in rs:
                rs["rope_type"] = "default"
            if "factor" not in rs:
                rs["factor"] = 1.0

    _require_keys({"seqorth_params": seqorth_params}, ["seqorth_params"], context="model_config")
    
    # Robustly get attn_implementation
    attn_from_cfg = seqorth_params.get("attn_implementation")
    if not attn_from_cfg:
        logger.warning("  ⚠️ seqorth_params.attn_implementation missing, defaulting to sdpa")
        attn_from_cfg = "sdpa"
    
    # Support pipe-separated options (e.g., "paged|flash_attention_3")
    # Extract the actual attention implementation (last part after pipe)
    if "|" in attn_from_cfg:
        attn_parts = [p.strip() for p in attn_from_cfg.split("|")]
        # Use the last part as the actual attention implementation
        attn_from_cfg = attn_parts[-1]
        attn_implementation = attn_from_cfg
        logger.info(f"  🔧 Using attn_implementation: {attn_implementation}")
    
    # Support flash_attention_3 (requires flash-attn-3 package and compatible hardware)
    valid_attn_implementations = {"eager", "sdpa", "flash_attention_2", "flash_attention_3"}
    if attn_from_cfg not in valid_attn_implementations:
        raise ValueError(f"Invalid seqorth_params.attn_implementation={attn_from_cfg}. Valid options: {valid_attn_implementations}")
    
    # Check flash_attention_3 availability if requested (use transformers' built-in check)
    if attn_from_cfg == "flash_attention_3":
        if is_flash_attn_3_available():
            logger.info("  ✅ flash-attn-3 package detected - using flash_attention_3")
        else:
            logger.warning("  ⚠️ flash-attn-3 package not available. Falling back to flash_attention_2")
            logger.info("  💡 To use flash_attention_3, install: pip install flash-attn-3")
            attn_from_cfg = "flash_attention_2"
    
    attn_implementation = attn_from_cfg


    # Seqorth configuration parameters from config file
    # NOTE: we intentionally validate + use only params actually consumed by the runtime model.
    # Required for the paper experiments + reproducibility (no silent defaults).
    required_seqorth_params = [
        "n_shared_experts",
        "n_routed_experts",
        "num_experts_per_tok",
    ]
    _require_keys(seqorth_params, required_seqorth_params, context="model_config.seqorth_params")
    
    # Load and configure Seqorth model configuration
    if initialize_from_scratch:
        logger.info("Initializing model from scratch...")
        # Load base model config from tokenizer path to get actual architecture
        base_model_path = model_config.get("tokenizer_name_or_path") or model_config.get("model_name_or_path")
        if base_model_path:
            logger.info(f"📐 Loading base model architecture from: {base_model_path}")
            try:
                base_config = AutoConfig.from_pretrained(
                    base_model_path,
                    trust_remote_code=model_config["trust_remote_code"]
                )
                base_model_config = base_config.to_dict()
                
                # Handle different model config structures
                if 'text_config' in base_model_config:
                    text_config = base_model_config['text_config']
                else:
                    text_config = base_model_config
                
                # Get architecture parameters from base model config (must exist; no fallbacks)
                required_arch = ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size"]
                for k in required_arch:
                    if k not in text_config:
                        raise KeyError(f"Base model text_config missing required key: {k}")

                hidden_size = text_config["hidden_size"]
                num_hidden_layers = text_config["num_hidden_layers"]
                num_attention_heads = text_config["num_attention_heads"]
                num_key_value_heads = text_config.get("num_key_value_heads", num_attention_heads)
                intermediate_size = text_config["intermediate_size"]
                vocab_size = text_config["vocab_size"]
                max_position_embeddings = text_config.get("max_position_embeddings")
                if max_position_embeddings is None:
                    raise KeyError("Base model text_config missing required key: max_position_embeddings")
                
                logger.info(f"  ✅ Loaded architecture: layers={num_hidden_layers}, hidden_size={hidden_size}, heads={num_attention_heads}")
            except Exception as e:
                logger.error(f"  ❌ Failed to load base model config for from-scratch init: {e}")
                raise
        else:
            raise ValueError("Cannot initialize from scratch without tokenizer_name_or_path or model_name_or_path")
        
        # Create a minimal config from scratch
        
        # Create text config from scratch (strictly driven by config + base arch; no hidden defaults)
        # Create text config from scratch (strictly driven by config + base arch; no hidden defaults)
        text_config_dict = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": hidden_size // num_attention_heads,
            "model_type": "seqorth_text",
            "attn_implementation": attn_implementation,
            "max_position_embeddings": max_position_embeddings,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
        }
        
        # Inject Seqorth parameters directly (concise)
        text_config_dict.update(seqorth_params)
        
        # Ensure default for router_impl if not present
        if "router_impl" not in text_config_dict:
            text_config_dict["router_impl"] = "seqorth"
        
        # Create vision config (minimal for from-scratch)
        vision_config = SiglipVisionConfig(
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
        )
        
        # Create Seqorth config from scratch
        config = SeqorthConfig(
            text_config=text_config_dict,
            vision_config=vision_config,
            boi_token_index=255999,
            eoi_token_index=256000,
            image_token_index=262144,
            initializer_range=0.02,
            attn_implementation=attn_implementation,
        )
    else:
        logger.info("Loading base model configuration...")
        base_config = AutoConfig.from_pretrained(
            model_config["model_name_or_path"],
            trust_remote_code=model_config["trust_remote_code"]
        )
        
        # Convert to dict and update with Seqorth parameters
        base_model_config = base_config.to_dict()
        
        # Handle different model config structures (Gemma vs others)
        if 'text_config' in base_model_config:
            # Multi-modal model with text_config
            text_config = base_model_config['text_config']
            num_attention_heads = text_config['num_attention_heads']
        else:
            # Direct text model config
            text_config = base_model_config
            num_attention_heads = base_model_config['num_attention_heads']
        
        # Core MoE/router params - directly update from seqorth_params
        base_model_config["text_config"].update(seqorth_params)
        
        # Ensure necessary overrides
        base_model_config["text_config"].update({
            "model_type": "seqorth_text",
            "attn_implementation": attn_implementation,
        })
        
        # Ensure default for router_impl if not present
        if "router_impl" not in base_model_config["text_config"]:
            base_model_config["text_config"]["router_impl"] = "seqorth"
        # Create Seqorth configuration
        config = SeqorthConfig(
            text_config=text_config,
            vision_config=base_model_config.get("vision_config"),
            boi_token_index=model_config.get("boi_token_index", base_model_config.get("boi_token_index", 0)),
            eoi_token_index=model_config.get("eoi_token_index", base_model_config.get("eoi_token_index", 0)),
            image_token_index=model_config.get("image_token_index", base_model_config.get("image_token_index", 0)),
            initializer_range=base_model_config.get("initializer_range", 0.02),
            attn_implementation=attn_implementation,
            # Fix: Match mm_tokens_per_image to Siglip output (14x14 = 196) to prevent stride=0 error
            mm_tokens_per_image=196,
            **{
                k:v for k,v in base_model_config.items() 
                if k not in [
                    "text_config", "vision_config", "boi_token_index",
                    "eoi_token_index", "image_token_index", "initializer_range",
                    "attn_implementation"
                ]
            }
        )
    
    # CRITICAL for NVMe/PEFT: Untie word embeddings to prevent swap buffer conflicts
    # This prevents "param already assigned swap buffer id" errors in DeepSpeed ZeRO-3
    config.tie_word_embeddings = False
    logger.info("  🔓 Explicitly set tie_word_embeddings = False for NVMe offloading compatibility")

    logger.info("Seqorth configuration created successfully")
    logger.info(f"  Shared experts: {config.text_config.n_shared_experts}, Routed experts: {config.text_config.n_routed_experts}, Experts per token: {config.text_config.num_experts_per_tok}")
    
    # CRITICAL: Ensure vision tower dtype matches the selected global model_dtype.
    # Some VLMs (e.g. Seqorth) default vision to FP16 which can mismatch bf16/fp32 runs.
    if hasattr(config, 'vision_config') and config.vision_config is not None:
        config.vision_config.dtype = model_dtype
        logger.debug(f"  ✅ Vision config dtype set to {model_dtype}")
    
    # Import deepspeed if needed
    # Load model - use different device_map strategy based on DeepSpeed usage
    device_map = None
    use_low_cpu_mem_usage = True
    
    if model_config.get("deepspeed_config"):
        # CRITICAL: DeepSpeed ZeRO handles parameter sharding, so device_map must be None
        # Using device_map="auto" causes CUDA unspecified launch failure
        device_map = None
        # DeepSpeed ZeRO with low_cpu_mem_usage=True helps reduce RAM spikes during load
        use_low_cpu_mem_usage = True
        logger.info("Using DeepSpeed - letting DeepSpeed handle device placement")
    elif torch.cuda.device_count() > 1 and os.environ.get("ACCELERATE_USE_FSDP", "false").lower() != "true" and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        # Only use auto device map for single-node non-distributed inference/naive training
        device_map = "auto"
        logger.info(f"Using auto device mapping for {torch.cuda.device_count()} GPUs (Single Process)")
    else:
        # For FSDP, DDP, or DeepSpeed, default to CPU load (device_map=None)
        # accelerator.prepare() will handle moving to GPU
        device_map = None
        logger.info("Using CPU loading (device_map=None) - letting FSDP/DDP handle device placement")
    
    # Load Seqorth model with the configured parameters
    logger.info("🤖 Loading Seqorth model...")
    if initialize_from_scratch:
        logger.info(f"🤖 Initializing from scratch (no pretrained model)")
    else:
        logger.info(f"🤖 Model path: {model_config.get('model_name_or_path', 'N/A')}")
    logger.info(f"🤖 Device map: {device_map}")
    logger.info(f"🤖 Attention implementation: {attn_implementation}")
    
    # Log memory before model loading
    memory_before = log_gpu_memory(logger, "BEFORE_MODEL_LOAD")
    
    # CRITICAL: With HfDeepSpeedConfig, model will be initialized directly on meta device
    # and weights will be loaded sharded across GPUs - NO CPU materialization needed
    # CRITICAL: device_map must be None with DeepSpeed - it conflicts with ZeRO sharding
    device_map = None  # Required for DeepSpeed - it handles placement
    max_memory = None
    logger.info("  🚀 Using HfDeepSpeedConfig for direct meta device initialization (no CPU load)")

    try:
        start_time = time.time()
        # Prepare DeepSpeed config (sanitize "auto" values for zero.Init)
        ds_config_dict = None
        if model_config.get("deepspeed_config"):
            ds_config_path = model_config.get("deepspeed_config")
            if isinstance(ds_config_path, str) and os.path.exists(ds_config_path):
                with open(ds_config_path, 'r') as f:
                    ds_config_dict = json.load(f)
            elif isinstance(ds_config_path, dict):
                ds_config_dict = ds_config_path.copy()
            else:
                ds_config_dict = {}

            # ZeRO-2 OOM fix: ensure offload_optimizer (nvme) is present so fp32 partition goes to CPU, not GPU
            if ds_config_dict and ds_config_dict.get("zero_optimization", {}).get("stage") == 2:
                zo = ds_config_dict["zero_optimization"]
                if not zo.get("offload_optimizer") or zo["offload_optimizer"].get("device") != "nvme":
                    zo["offload_optimizer"] = {
                        "device": "nvme",
                        "nvme_path": zo.get("offload_optimizer", {}).get("nvme_path") or "/mls/conan/offload",
                        "pin_memory": False,
                        "fast_init": True,
                        "ratio": 1,
                        "buffer_count": 8,
                        "pipeline_read": False,
                        "pipeline_write": False,
                    }
                nvme_path = zo["offload_optimizer"].get("nvme_path")
                if nvme_path:
                    try:
                        os.makedirs(nvme_path, exist_ok=True)
                        if is_main_process():
                            logger.info(f"  ✅ ZeRO-2 offload_optimizer nvme_path ensured: {nvme_path}")
                    except OSError as e:
                        logger.warning(f"  ⚠️ Could not create nvme_path {nvme_path}: {e}")

            # CRITICAL: Read SP and TP size from Accelerate config and apply to DeepSpeed config
            sp_size = 1
            tp_size = 1
            try:
                project_root = Path(__file__).parent.parent
                accelerate_config_path = project_root / "config" / "seqorth" / "accelerate.yaml"
                if accelerate_config_path.exists():
                    with open(accelerate_config_path, 'r') as f:
                        accelerate_config = yaml.safe_load(f)
                        parallelism_config = accelerate_config.get("parallelism_config", {})
                        sp_size = parallelism_config.get("sp_size", 1)
                        tp_size = parallelism_config.get("tp_size", 1)
                        if sp_size > 1:
                            ds_config_dict["sequence_parallel_size"] = sp_size
                            logger.info(f"✅ Applied SP size {sp_size} from Accelerate config to DeepSpeed config")
                        if "zero_optimization" not in ds_config_dict:
                            ds_config_dict["zero_optimization"] = {}
                        zero_opt = ds_config_dict["zero_optimization"]                        
                        
                        # Use configuration from JSON/Accelerate directly without script-side overrides
                        # to ensure consistency with test_seqorth_peft behavior.
                        logger.info("✅ Relying on JSON/Accelerate config for DeepSpeed Buffer and NVMe settings")
                        # --------------------------------------------------------------------------

                        if tp_size > 1:
                            # Check if autotp is already configured
                            if "tensor_parallel" in ds_config_dict and "autotp_size" in ds_config_dict.get("tensor_parallel", {}):
                                logger.info(f"⚠️  autotp_size already set in DeepSpeed config, skipping tensor_model_parallel_size from Accelerate")
                                
                                # CRITICAL: Set injection policy to exclude embedding layers
                                tensor_parallel_config = ds_config_dict.get("tensor_parallel", {})
                                if "tp_injection_policy" not in tensor_parallel_config:
                                    logger.info(f"  ℹ️  Will set tp_injection_policy to exclude embedding layers during DeepSpeed init")
                            else:
                                # CRITICAL: TP must be set in DeepSpeed config for ZeRO-3 to recognize TP groups
                                ds_config_dict["tensor_model_parallel_size"] = tp_size
                                logger.info(f"✅ Applied TP size {tp_size} from Accelerate config to DeepSpeed config")

                            logger.info(f"✅ ZeRO-3 configured to work with TP size {tp_size}")
                        
                        # Rely on JSON config for stage3_param_persistence_threshold - no script-side override

            except Exception as e:
                logger.warning(f"⚠️ Failed to read parallelism config from Accelerate config: {e}, using defaults (SP=1, TP=1)")

            # CRITICAL: Calculate train_batch_size BEFORE HfDeepSpeedConfig initialization
            # Match light_seqorth.py: calculate actual values from training config
            # Use global dist import (already imported at top of file)
            try:
                world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))
            except:
                # Fallback if dist not initialized - use WORLD_SIZE env var
                world_size = int(os.environ.get("WORLD_SIZE", 1))
            logger.info(f"  📊 World size for batch calculation: {world_size}")
            per_device = training_config.get("per_device_train_batch_size", 1)
            grad_accum = training_config.get("gradient_accumulation_steps", 1)
            
            # CRITICAL: Removing train_batch_size from config allows DeepSpeed to calculate it automatically
            # based on whatever world_size it detects at runtime. This bypasses the strict assertion check
            # that fails when different ranks see different world sizes during initialization.
            if "train_batch_size" in ds_config_dict:
                logger.info("  🗑️  Removing train_batch_size from DeepSpeed config to bypass assertion checks")
                del ds_config_dict["train_batch_size"]
            
            # Ensure micro_batch and grad_accum are set (defaults to 1 if missing)
            if ds_config_dict.get("train_micro_batch_size_per_gpu") == "auto":
                ds_config_dict["train_micro_batch_size_per_gpu"] = per_device
            if "train_micro_batch_size_per_gpu" not in ds_config_dict:
                 ds_config_dict["train_micro_batch_size_per_gpu"] = per_device

            if ds_config_dict.get("gradient_accumulation_steps") == "auto":
                ds_config_dict["gradient_accumulation_steps"] = grad_accum
            if "gradient_accumulation_steps" not in ds_config_dict:
                ds_config_dict["gradient_accumulation_steps"] = grad_accum
            
            # Patch optimizer params for "auto" values - use training config values
            if "optimizer" in ds_config_dict and "params" in ds_config_dict["optimizer"]:
                for k in ["lr", "weight_decay"]:
                    if ds_config_dict["optimizer"]["params"].get(k) == "auto":
                        if k == "lr":
                            ds_config_dict["optimizer"]["params"][k] = training_config.get("learning_rate", 1e-5)
                        elif k == "weight_decay":
                            ds_config_dict["optimizer"]["params"][k] = training_config.get("weight_decay", 0.0)
            
            # Adjust batch size for SP if enabled (only if SP > 1)
            if sp_size > 1 and "train_batch_size" in ds_config_dict and isinstance(ds_config_dict["train_batch_size"], int):
                original_batch = ds_config_dict["train_batch_size"]
                if original_batch % sp_size == 0:
                    ds_config_dict["train_batch_size"] = original_batch // sp_size
                    logger.info(f"✅ [SP] Adjusted train_batch_size for SP: {original_batch} -> {ds_config_dict['train_batch_size']} (sp_size={sp_size})")
            
            # CRITICAL: Save modified config to temporary file for HfDeepSpeedConfig
            # Match light_seqorth.py: use temp file to ensure HfDeepSpeedConfig reads the modified config
            temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(ds_config_dict, temp_config_file, indent=2)
            temp_config_file.close()
            temp_config_path = temp_config_file.name
            logger.info(f"  💾 Saved modified DeepSpeed config to temporary file: {temp_config_path}")
            
            # CRITICAL: Set PYTORCH_CUDA_ALLOC_CONF to avoid fragmentation OOM
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # CRITICAL: Export to environment so DeepSpeed and Transformers can find it
            os.environ["DEEPSPEED_CONFIG_FILE"] = temp_config_path
            logger.info(f"  📤 Exported DEEPSPEED_CONFIG_FILE={temp_config_path}")
            
            # Update model_config to use temporary config file
            model_config["deepspeed_config"] = temp_config_path

        if initialize_from_scratch:
            # Initialize model from scratch with random weights
            logger.info("🔨 Initializing model from scratch (random weights)...")
            if ds_config_dict:
                logger.info("  ⚡ Using DeepSpeed ZeRO Init for efficient initialization")
                
                # Eagerly initialize vision tower (small) to avoid ZeRO-3 Init crash (SigLIP fan-in issue with sharded weights)
                logger.info("  ⚡ Eagerly initializing vision tower to avoid ZeRO-3 Init compatibility issues...")
                
                # Memory optimization: clear cache before vision tower init
                gc.collect()
                torch.cuda.empty_cache()
                
                vision_tower = AutoModel.from_config(config=config.vision_config, trust_remote_code=True)
                
                # Clear cache after vision tower init
                gc.collect()
                torch.cuda.empty_cache()

                with deepspeed.zero.Init(config_dict_or_path=ds_config_dict):
                    model = SeqorthForConditionalGeneration(config=config, vision_tower=vision_tower)
                
                # Clear cache after model init
                gc.collect()
                torch.cuda.empty_cache()
            else:
                model = SeqorthForConditionalGeneration(config=config)
            
            # Defer device/dtype placement to PEFT/Trainer to avoid multi-GPU OOM
        else:
            # [ZeRO Fix] Universal Exoskeleton Loading Path
            # This path loads the base model directly to ensure 100% pretrained weight alignment,
            # then injects Seqorth MoE components into the existing structure.
            zero_stage_preview = ds_config_dict.get("zero_optimization", {}).get("stage", 0) if ds_config_dict else 0
            logger.info(f"  ⚡ Loading base model with ZeRO-{zero_stage_preview} setup...")
            
            # Additional memory optimization: clear GPU memory before loading
            clear_gpu_memory()
            
            # Local snapshot resolution (to avoid Hub check failures)
            model_id = model_config["model_name_or_path"]
            if not os.path.exists(model_id):
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                repo_folder = f"models--{model_id.replace('/', '--')}"
                full_repo_path = os.path.join(cache_dir, repo_folder)
                if os.path.isdir(full_repo_path):
                    snapshots_path = os.path.join(full_repo_path, "snapshots")
                    if os.path.isdir(snapshots_path):
                        snapshots = os.listdir(snapshots_path)
                        if snapshots:
                            model_id = os.path.join(snapshots_path, snapshots[0])
                            logger.info(f"📍 Resolved repo ID to local snapshot: {model_id}")
            
            logger.info(f"🚀 Loading base model from: {model_id}")
            
            # ZeRO stage: only ZeRO-3 supports meta+partition; ZeRO-2 cannot use zero.Init (meta tensor cannot .to(device) without partition materialize)
            zero_stage = ds_config_dict.get("zero_optimization", {}).get("stage", 0) if ds_config_dict else 0

            if zero_stage == 3:
                # [ZeRO-3 Fix] Initialize HfDeepSpeedConfig and patch remote modules BEFORE model loading
                hf_ds_config = HfDeepSpeedConfig(ds_config_dict) if ds_config_dict else None
                
                # [ZeRO-3 Fix] Aggressively apply Universal Guard to identify and protect 
                # all Block/Layer/Merger/MLP modules across current modeling classes.
                try:
                    # 1. Force pre-load of remote classes to ensure they are available in sys.modules
                    tmp_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                    from transformers.dynamic_module_utils import get_class_from_dynamic_module
                    if hasattr(tmp_config, "auto_map"):
                        for key in ["AutoModelForVision2Seq", "AutoModel"]:
                            if key in tmp_config.auto_map:
                                class_ref = tmp_config.auto_map[key]
                                get_class_from_dynamic_module(class_ref, model_id)
                                break
                    
                    # 2. Apply Universal Guard (Pattern-based discovery)
                    apply_universal_exoskeleton_guard(logger)
                    logger.info("✅ Applied Universal Exoskeleton Guard to fix size 0 mismatch issues.")
                except Exception as e:
                    logger.warning(f"⚠️ Universal Guard application failed: {e}")

                dist.barrier()
                
                # Load model within ZeRO-3 Init context (meta device)
                logger.info("  ⚡ Using ZeRO-3 meta device initialization...")
                init_context = deepspeed.zero.Init(config_dict_or_path=ds_config_dict)
            else:
                # ZeRO-2 or no ZeRO: regular load (zero.Init causes "Cannot copy out of meta tensor" - ZeRO-2 does not partition params)
                logger.info(f"  ⚡ Using ZeRO-{zero_stage} regular loading (no meta device)...")
                init_context = contextlib.nullcontext()  # No-op context manager
            
            with init_context:
                # 1. Load the BASE MODEL directly (aligned keys!)
                # [ZeRO-3 Fix] Qwen-VL-MoE (Multimodal) must be loaded using AutoModelForVision2Seq or AutoModel.
                # AutoModelForCausalLM rejects Qwen3VLMoeConfig as it is not strictly a CausalLM class.
                logger.info("  ⚡ Instantiating base model (searching for correct AutoModel class)...")
                
                try:
                    # Qwen-VL and other VLMs usually map to AutoModelForVision2Seq
                    base_model = AutoModelForVision2Seq.from_pretrained(
                        model_id,
                        dtype=model_dtype,
                        trust_remote_code=model_config["trust_remote_code"],
                        device_map=None, 
                        low_cpu_mem_usage=True,
                        offload_state_dict=True,
                        use_safetensors=True,
                        attn_implementation=attn_implementation,
                        ignore_mismatched_sizes=True, 
                    )
                    logger.info("  ✅ Loaded base model successfully using AutoModelForVision2Seq")
                except Exception as v_e:
                    logger.warning(f"  ⚠️ AutoModelForVision2Seq not compatible or failed: {v_e}")
                    logger.info("  ⚡ Attempting load via AutoModel (generic fallback)...")
                    base_model = AutoModel.from_pretrained(
                        model_id,
                        dtype=model_dtype,
                        trust_remote_code=model_config["trust_remote_code"],
                        device_map=None, 
                        low_cpu_mem_usage=True,
                        offload_state_dict=True,
                        use_safetensors=True,
                        attn_implementation=attn_implementation,
                        ignore_mismatched_sizes=True, 
                    )
                    logger.info("  ✅ Loaded base model successfully using AutoModel")
                
                # 2. Universal Exoskeleton MoE Injection (또는 baseline 시 건너뜀)
                if os.environ.get("SEQORTH_DISABLE_ALL", "0") == "1":
                    logger.info("  🔓 [baseline] Seqorth injection 건너뜀 → base_model 그대로 사용")
                    model = base_model
                else:
                    # Replace Qwen3/Base MLPs with Seqorth advanced routing mechanism
                    logger.info("  🔧 Injecting Seqorth Exoskeleton (OSR/LDR Router Swap)...")
                    global_router = SeqorthRouter(config.text_config)
                    injector = SeqorthExoskeletonMoEInjector(config.text_config, global_router)
                    model = injector.inject(base_model)
                
                # Optional: Ensure image features handled via no_grad logic if needed
                # (Qwen3 vision tower can be huge, freezing it is standard)
                model_load_result = model
                
            model = model_load_result
                
            # Unwrap return values
            if isinstance(model_load_result, tuple):
                model = model_load_result[0]
                loading_info = model_load_result[1]
                logger.info(f"  ⚠️ Loading Info - Missing Keys: {len(loading_info.get('missing_keys', []))}")
                if len(loading_info.get('missing_keys', [])) > 0:
                    logger.warning(f"  Missing Keys Sample: {loading_info['missing_keys'][:20]}")
                if len(loading_info.get('unexpected_keys', [])) > 0:
                    logger.warning(f"  Unexpected Keys Sample: {loading_info['unexpected_keys'][:20]}")
            else:
                model = model_load_result


            dschf = None
            
            # 모델 로드 완료
            logger.info(" ✅ Model loaded.")
    except Exception as e:
        logger.error(f"❌ Failed to load Seqorth model: {e}")
        raise e
    
    # Ensure router AND expert parameters are trainable
    # CRITICAL FIX: Force initialization of new components (priority_head, mm_projector, etc.)
    # These might be NaN/garbage if loaded with strict=False from a checkpoint that lacks them.
    
    actual_model_for_init = model.module if hasattr(model, 'module') else model
    
    # 0. Fix Multi-Modal Projector (new component, not in checkpoint)
    logger.info("🔧 Initializing Multi-Modal Projector...")
    if hasattr(actual_model_for_init, 'model') and hasattr(actual_model_for_init.model, 'multi_modal_projector'):
        mm_proj = actual_model_for_init.model.multi_modal_projector
        for name, param in mm_proj.named_parameters():
            # [수정] 데이터가 있는 경우(param.numel() > 0)에만 값을 검사하도록 논리적으로 결합
            # 빈 텐서는 NaN이나 Inf가 될 수 없으므로 검사할 필요가 없습니다.
            if param.numel() > 0 and (torch.isnan(param).any() or torch.isinf(param).any() or param.abs().max() > 1e10):
                logger.warning(f"  ⚠️ MM Projector {name} is invalid, reinitializing...")
                if 'norm' in name and 'weight' in name:
                    nn.init.zeros_(param)
                elif param.ndim >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
        logger.info("  ✅ MM Projector initialized.")
    
    logger.info("🔧 Verifying and Initializing Router Components...")
    actual_model_for_init = model.module if hasattr(model, 'module') else model
    fixed_routers = 0
    fixed_grus = 0
    for name, module in actual_model_for_init.named_modules():
        if "SeqorthRouter" in type(module).__name__:
            # 1. Fix Priority Head
            if hasattr(module, 'priority_head') and isinstance(module.priority_head, nn.Linear):
                nn.init.zeros_(module.priority_head.weight)
                fixed_routers += 1
            
            # 1.5 Fix Bias Prediction FC layers (new components)
            for fc_name in ['bias_pred_fc1', 'bias_pred_fc2']:
                if hasattr(module, fc_name):
                    fc = getattr(module, fc_name)
                    if isinstance(fc, nn.Linear):
                        nn.init.xavier_uniform_(fc.weight)
                        if fc.bias is not None:
                            nn.init.zeros_(fc.bias)
            
            # 2. Fix Load Balancer (GRU) - Missing in checkpoint -> NaN
            if hasattr(module, 'load_balancer'):
                gru = module.load_balancer
                for layer_name in ['weight_ih_gates', 'weight_hh_gates', 'weight_ih_cand', 'weight_hh_cand', 'bias_ih_gates', 'bias_hh_gates', 'bias_ih_cand', 'bias_hh_cand']:
                     if hasattr(gru, layer_name):
                         attr = getattr(gru, layer_name)
                         
                         # Case 1: Attribute is nn.Linear (Custom Implementation)
                         if isinstance(attr, nn.Linear):
                             nn.init.orthogonal_(attr.weight)
                             if attr.bias is not None:
                                 nn.init.zeros_(attr.bias)
                             fixed_grus += 1
                             
                         # Case 2: Attribute is Tensor (Standard GRUCell / Parameter)
                         elif isinstance(attr, torch.Tensor) or isinstance(attr, nn.Parameter):
                             try:
                                 if 'bias' in layer_name:
                                     nn.init.zeros_(attr)
                                 else:
                                     if attr.ndim >= 2:
                                         nn.init.orthogonal_(attr)
                                     else:
                                         # Fallback for 1D/Partitioned weights: Use Normal instead of Zero/Xavier
                                         nn.init.normal_(attr, std=0.02)
                             except ValueError:
                                  # If orthogonal fails (dimension issue), fallback to normal
                                  if 'bias' in layer_name:
                                      nn.init.zeros_(attr)
                                  else:
                                      nn.init.normal_(attr, std=0.02)
                                      
                             fixed_grus += 1

    if fixed_routers > 0:
        logger.info(f"  ✅ Force-initialized {fixed_routers} priority_heads to Zeros.")
    if fixed_grus > 0:
        logger.info(f"  ✅ Force-initialized {fixed_grus} GRU weight matrices to Orthogonal.")
    
    # 3. Fix new LayerNorm components (not in checkpoint)
    logger.info("🔧 Initializing new LayerNorm components...")
    fixed_layernorms = 0
    for name, module in actual_model_for_init.named_modules():
        if "SeqorthDecoderLayer" in type(module).__name__:
            for ln_name in ['pre_feedforward_layernorm', 'post_feedforward_layernorm']:
                if hasattr(module, ln_name):
                    ln = getattr(module, ln_name)
                    if hasattr(ln, 'weight') and torch.isnan(ln.weight).any():
                        nn.init.ones_(ln.weight)
                        fixed_layernorms += 1
    
    if fixed_layernorms > 0:
        logger.info(f"  ✅ Force-initialized {fixed_layernorms} new LayerNorms to Ones.")
    
    router_params, router_names, _ = ensure_router_parameters_trainable(model, logger, context="setup_model")
    
    # CRITICAL: Also ensure experts are trainable for MoE/Upcycling
    logger.info("🔧 Ensuring all experts (SeqorthMLP) are trainable...")
    experts_trainable_count = 0
    actual_model = model.module if hasattr(model, 'module') else model
    for name, module in actual_model.named_modules():
        if "SeqorthMLP" in type(module).__name__ or "SeqorthExpert" in type(module).__name__:
            for p in module.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    experts_trainable_count += p.numel()
    
    if experts_trainable_count > 0:
        logger.info(f"✅ Experts made trainable: {experts_trainable_count} params")
    modules_to_save_list_to_return = None
    apply_seqorth_patch(model, logger)
    return model, modules_to_save_list_to_return
def main(
    model_config: Dict[str, Any], 
    data_config: Dict[str, Any], 
    training_config: Dict[str, Any],
    config_path: str = None
):
    # [baseline] SEQORTH_DISABLE_ALL=1 → Seqorth injection 건너뜀
    _seqorth_disable_all = os.environ.get("SEQORTH_DISABLE_ALL", "0") == "1"
    if _seqorth_disable_all:
        logger.info("🔓 [baseline] SEQORTH_DISABLE_ALL=1 → Seqorth injection 건너뜀 (Qwen3 원래 MoE만 사용)")
        # baseline 시 ZeRO-2 사용: ZeRO-3 파티션 시 backward 'tensor a (0) vs b (2048)' 방지
        _ds = model_config.get("deepspeed_config")
        if _ds and isinstance(_ds, str) and os.path.exists(_ds):
            _dir = os.path.dirname(os.path.abspath(_ds))
            _z2 = os.path.join(_dir, "deepspeed_zero2.json")
            if os.path.exists(_z2):
                model_config["deepspeed_config"] = _z2
                if training_config.get("deepspeed"):
                    training_config["deepspeed"] = _z2
                logger.info(f"🔓 [baseline] DeepSpeed ZeRO-2 사용: {_z2}")
    
    # 경고를 로깅으로 캡처
    def warning_to_logging(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"⚠️  {category.__name__}: {message} (at {filename}:{lineno})")
        sys.stdout.flush()
        sys.stderr.flush()
    
    warnings.showwarning = warning_to_logging
    
    # CRITICAL: Pin each process to its GPU so NCCL does not see "Duplicate GPU"
    # launcher(accelerate/torchrun)가 CUDA_VISIBLE_DEVICES에 맞게 LOCAL_RANK를 설정해야 함
    # 우리는 launcher가 설정한 LOCAL_RANK를 그대로 사용 (강제 매핑 금지)
    if torch.cuda.is_available():
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
            num_visible_gpus = torch.cuda.device_count()
            if 0 <= local_rank < num_visible_gpus:
                torch.cuda.set_device(local_rank)
            else:
                # LOCAL_RANK가 CUDA_VISIBLE_DEVICES 범위를 벗어남 - launcher 설정 문제
                raise ValueError(
                    f"LOCAL_RANK={local_rank} >= num_visible_gpus={num_visible_gpus}. "
                    f"Launcher가 CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}에 맞게 LOCAL_RANK를 설정해야 합니다. "
                    f"예: CUDA_VISIBLE_DEVICES=0,1이고 4프로세스면 각 노드의 LOCAL_RANK는 0,1,0,1이어야 함."
                )
    
    # CRITICAL FIX: Disable HF gradient checkpointing if FSDP is enabled
    if os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true":
        if training_config.get("gradient_checkpointing"):
            logger.warning("⚠️  FSDP detected: Forcing training_config['gradient_checkpointing'] = False")
            training_config["gradient_checkpointing"] = False
    
    # ===== DEBUGGING: Respect DETECT_ANOMALY=1 for 0 vs 2048 backward traceback =====
    if os.environ.get("DETECT_ANOMALY", "0") == "1":
        torch.autograd.set_detect_anomaly(True)
        logger.warning("🐛 DETECT_ANOMALY=1 → autograd anomaly ON (에러 시 backward 연산 위치 표시)")
    else:
        torch.autograd.set_detect_anomaly(False)
        logger.info("ℹ️ Autograd anomaly detection DISABLED")
    
    # Initialize distributed environment manually if needed
    # This ensures DeepSpeed can detect world size correctly during HfDeepSpeedConfig initialization
    # CRITICAL: Pass device_id so NCCL uses the correct GPU per rank (avoids "calloc async 608 bytes" / Duplicate GPU)
    # CRITICAL: ZeRO-3 all_gather (NVMe offload) can be slow; 60min timeout to avoid Watchdog. No fallback.
    if not dist.is_initialized():
        try:
            from datetime import timedelta
            _nccl_timeout = timedelta(minutes=60)
            local_rank_for_init = int(os.environ.get("LOCAL_RANK", 0))
            init_kwargs = {"backend": "nccl", "timeout": _nccl_timeout}
            if torch.cuda.is_available() and 0 <= local_rank_for_init < torch.cuda.device_count():
                init_kwargs["device_id"] = local_rank_for_init
            dist.init_process_group(**init_kwargs)
            logger.info("✅ Manually initialized distributed process group (with device_id for NCCL, timeout=60min)")
        except TypeError:
            from datetime import timedelta
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
            logger.info("✅ Manually initialized distributed process group (timeout=60min)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to manually initialize distributed process group: {e}")
            pass
    
    # 1. Setup tokenizer FIRST
    logger.debug("Setting up tokenizer...")
    tokenizer = setup_tokenizer(model_config)
    
    # 2. Setup model (Heavy memory usage) - DO THIS FIRST to reserve GPU/CPU RAM for model
    logger.debug("Setting up model...")
    setup_result = setup_model(model_config)
    if isinstance(setup_result, tuple) and len(setup_result) == 2:
        model, modules_to_save_list = setup_result
    else:
        model = setup_result
        modules_to_save_list = None
    
    # Aggressive memory cleanup after model load
    gc.collect()
    torch.cuda.empty_cache()
    
    # === CRITICAL: Apply PEFT LoRA (Missing in original script) ===
    if model_config.get("use_lora", False):
        logger.info("🚀 [PEFT] Applying LoRA wrapping...")
        
        # 1. Identify target modules for LoRA
        target_modules = get_dynamic_lora_target_modules(model, logger)
        
        # 2. Setup LoraConfig (WITHOUT modules_to_save for experts to avoid OOM)
        # We will manually unfreeze experts/routers later
        peft_config = LoraConfig(
            r=model_config.get("lora_r", 16),
            lora_alpha=model_config.get("lora_alpha", 32),
            target_modules=target_modules,
            lora_dropout=model_config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=[] # Empty to prevent ModulesToSaveWrapper OOM
        )
        
        # Aggressive memory cleanup before PEFT wrapping
        gc.collect()
        torch.cuda.empty_cache()
        
        model = get_peft_model(model, peft_config)
        logger.info("✅ [PEFT] LoRA wrapping applied successfully")

        # 3. Manually unfreeze Experts and Routers (Hybrid Training: LoRA + FFT)
        logger.info("🔓 [PEFT] Manually unfreezing Experts and Routers for FFT...")
        actual_model = model.module if hasattr(model, 'module') else model
        count_unfrozen = 0
        frozen_modules = []
        
        # Identify modules to unfreeze
        # Note: 'base_layer' in LoRA wrapped modules might be frozen by get_peft_model
        # But since we excluded experts from LoRA targets, they are NOT wrapped, so they are just frozen standard modules.
        # We simply need to set requires_grad=True.
        for name, module in actual_model.named_modules():
            class_name = type(module).__name__
            if any(k in class_name for k in ["SeqorthMLP", "SeqorthExpert", "SeqorthRouter", "Qwen3VLMoeTextExperts", "Qwen3VLMoeTextSparseMoeBlock"]):
                for param in module.parameters():
                    param.requires_grad = True
                count_unfrozen += 1
                
        logger.info(f"✅ [PEFT] Unfrozen {count_unfrozen} MoE/Router modules for Full Fine-Tuning")
        
        # Re-check trainable parameters after LoRA
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 [PEFT] Post-LoRA Trainable: {trainable_params/1e9:.2f}B / {total_params/1e9:.2f}B ({(trainable_params/total_params)*100:.2f}%)")
    else:
        logger.info("🔓 [PEFT] LoRA disabled, performing Full Fine-Tuning (FFT)")

    # Aggressive memory cleanup after LoRA
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Setup dataset (Heavy processing) - Load after model to use remaining RAM
    logger.debug("Setting up dataset (Post-Model Load)...")
    dataset, collate_fn = setup_dataset(data_config, tokenizer, logger, training_config)
    
    # CRITICAL: Sync all ranks after dataset load to prevent NCCL timeout
    # (Rank 0 can be much slower in multi-domain dataset "균등화된 파일 검증"; others would wait at first collective and timeout)
    if dist.is_initialized():
        dist.barrier()
        logger.info("✅ All ranks finished dataset load (barrier passed).")
    
    # Final cleanup before training starts
    gc.collect()
    torch.cuda.empty_cache()
    
    # Verify MoE classes in model for DeepSpeed registration
    moe_layers_found = []
    moe_class_names_found = set()
    peft_wrapped_count = 0
    
    # Get actual model (handle PEFT wrapper)
    actual_model = model
    if hasattr(model, 'base_model'):
        actual_model = model.base_model.model if hasattr(model.base_model, 'model') else model.base_model
        logger.debug("  🔍 Using base_model for module detection (PEFT wrapper detected)")
    
    # Find all MoE modules in the model (including original MoE classes from pretrained model)
    # CRITICAL: Also check PEFT wrapper internal modules
    for name, module in actual_model.named_modules():
        detected_module = None
        detected_class_name = None
        is_peft_wrapped = False
        
        # 1. Check if this is a PEFT wrapper containing a MoE module
        if ModulesToSaveWrapper is not None and isinstance(module, ModulesToSaveWrapper):
            # Check modules_to_save dictionary for wrapped modules
            if hasattr(module, 'modules_to_save'):
                for adapter_name, inner_module in module.modules_to_save.items():
                    if isinstance(inner_module, SeqorthMoE):
                        detected_module = inner_module
                        detected_class_name = type(inner_module).__name__
                        is_peft_wrapped = True
                        peft_wrapped_count += 1
                        logger.debug(f"  ✅ Found PEFT-wrapped SeqorthMoE: {name} (adapter: {adapter_name})")
                        break
                    elif hasattr(inner_module, 'experts') and (hasattr(inner_module, 'router') or hasattr(inner_module, 'gate')):
                        class_name = type(inner_module).__name__
                        if any(keyword in class_name.lower() for keyword in ['moe', 'expert', 'sparse']):
                            detected_module = inner_module
                            detected_class_name = class_name
                            is_peft_wrapped = True
                            peft_wrapped_count += 1
                            logger.debug(f"  ✅ Found PEFT-wrapped MoE: {name} ({class_name}, adapter: {adapter_name})")
                            break
        
        # 2. Direct check for SeqorthMoE
        if detected_module is None and isinstance(module, SeqorthMoE):
            detected_module = module
            detected_class_name = type(module).__name__
        
        # 3. Check for other MoE patterns (experts + router/gate)
        if detected_module is None and hasattr(module, 'experts') and (hasattr(module, 'router') or hasattr(module, 'gate')):
            class_name = type(module).__name__
            if any(keyword in class_name.lower() for keyword in ['moe', 'expert', 'sparse']):
                detected_module = module
                detected_class_name = class_name
        
        # Record detected module
        if detected_module is not None:
            if is_peft_wrapped:
                moe_layers_found.append((name, detected_class_name, "PEFT-wrapped"))
            else:
                moe_layers_found.append((name, detected_class_name) if detected_class_name != "SeqorthMoE" else name)
            moe_class_names_found.add(detected_class_name)
    
    # Also check the original model (not just base_model) for any additional modules
    if actual_model is not model:
        logger.debug("  🔍 Also scanning original model (with PEFT wrapper) for additional modules...")
        for name, module in model.named_modules():
            if isinstance(module, SeqorthMoE):
                # Check if already found in base_model
                already_found = any(
                    (isinstance(item, tuple) and item[0] == name) or item == name 
                    for item in moe_layers_found
                )
                if not already_found:
                    moe_layers_found.append(name)
                    moe_class_names_found.add(type(module).__name__)
                    logger.debug(f"  ✅ Found additional SeqorthMoE in wrapper: {name}")
    
    logger.info(f"✅ Found {len(moe_layers_found)} MoE layers in model")
    if peft_wrapped_count > 0:
        logger.info(f"   📦 {peft_wrapped_count} MoE layers are PEFT-wrapped")
    if moe_layers_found:
        logger.info(f"   All MoE layers ({len(moe_layers_found)}):")
        for i, item in enumerate(moe_layers_found[:5]):  # Show first 5
            if isinstance(item, tuple):
                if len(item) == 3 and item[2] == "PEFT-wrapped":
                    logger.info(f"     [{i}] {item[0]} ({item[1]}) [PEFT-wrapped]")
                else:
                    logger.info(f"     [{i}] {item[0]} ({item[1]})")
            else:
                logger.info(f"     [{i}] {item}")
        if len(moe_layers_found) > 5:
            logger.info(f"     ... and {len(moe_layers_found) - 5} more")
        logger.info(f"   MoE class names found: {moe_class_names_found}")
    else:
        logger.warning("⚠️ No MoE layers found! DeepSpeed may fail to find MoE classes.")
        # Enhanced debugging information
        logger.warning("  🔍 Debugging: Checking model structure...")
        if hasattr(model, 'base_model'):
            logger.warning(f"     - Model has base_model: {type(model.base_model).__name__}")
        if hasattr(model, 'module'):
            logger.warning(f"     - Model has module attribute: {type(model.module).__name__}")
        
        # List first 20 module names for debugging
        sample_modules = []
        for i, (name, module) in enumerate(model.named_modules()):
            if i < 20:
                sample_modules.append(f"{name} ({type(module).__name__})")
        if sample_modules:
            logger.warning(f"     - First 20 modules: {sample_modules}")
        
        # Check if PEFT is applied
        if hasattr(model, 'peft_config'):
            logger.warning(f"     - PEFT is applied: {list(model.peft_config.keys())}")
            logger.warning("     - Try checking base_model for MoE modules")
    
    # Store MoE class names for DeepSpeed registration
    # Include both SeqorthMoE and any detected original MoE classes
    detected_moe_classes = list(moe_class_names_found) if moe_class_names_found else ["SeqorthMoE"]
    # Always include SeqorthMoE in case router swap creates it
    if "SeqorthMoE" not in detected_moe_classes:
        detected_moe_classes.append("SeqorthMoE")
    
    # Setup dataset - ALREADY DONE ABOVE
    logger.info("Dataset already setup.")
    
    # 모델 및 데이터셋 로드 후 메모리 정리
    logger.info("🧹 모델 및 데이터셋 로드 후 GPU 메모리 정리...")
    clear_gpu_memory(logger)
    
    # [DeepSpeed Fix] Disable automatic placement of model on device by Trainer.
    # When using DeepSpeed (especially with TP or large models), we must let DeepSpeed 
    # handle the move/partitioning from CPU to GPU. Moving the full model first causes OOM.
    if model_config.get("deepspeed_config"):
        training_config["place_model_on_device"] = False
        logger.info("🔧 [DeepSpeed] Requested Trainer to skip automatic model placement")

    # Create training arguments
    training_args = create_training_args(
        training_config, 
        model_config.get("deepspeed_config"),
        logger=logger
    )
    
    # Optional: build bnb 8bit optimizer when requested in DeepSpeed config (DeepSpeed does not create it from JSON)
    optimizer_for_trainer = None
    try:
        ds_cfg_path = model_config.get("deepspeed_config")
        if ds_cfg_path and os.path.exists(os.path.abspath(ds_cfg_path)):
            with open(os.path.abspath(ds_cfg_path), "r", encoding="utf-8") as f:
                ds_cfg = json.load(f)
            opt_block = ds_cfg.get("optimizer") or {}
            opt_type = str(opt_block.get("type", "")).strip().lower()
            bnb_8bit_types = ("adamw8bit", "adamw_8bit", "adamw_bnb_8bit", "pagedadamw8bit", "adam8bit", "pagedadam8bit")
            if opt_type in bnb_8bit_types:
                import bitsandbytes as bnb
                params = opt_block.get("params", {})
                lr = params.get("lr", params.get("learning_rate", "auto"))
                if lr == "auto":
                    lr = training_config.get("learning_rate", 2e-5)
                weight_decay = params.get("weight_decay", 0.01)
                betas = params.get("betas", (0.9, 0.999))
                if opt_type in ("pagedadamw8bit", "pagedadam8bit"):
                    opt_class = getattr(bnb.optim, "PagedAdamW8bit", None) or getattr(bnb.optim, "AdamW8bit")
                else:
                    opt_class = getattr(bnb.optim, "AdamW8bit", None) or getattr(bnb.optim, "Adam8bit", None)
                if opt_class is not None:
                    trainable_params = _get_trainable_parameters(model)
                    optimizer_for_trainer = opt_class(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
                    logger.info(f"  ✓ Using bnb 8bit optimizer: {opt_class.__name__} (lr={lr})")
    except Exception as bnb_e:
        logger.debug(f"  bnb 8bit optimizer not used: {bnb_e}")

    # Setup trainer
    logger.info("Setting up trainer...")
    
    # CRITICAL: Register SeqorthMoE with Accelerate/DeepSpeed before Trainer initialization
    # This prevents "Could not find a transformer layer class called 'SeqorthMoE'" error
    if model_config.get("deepspeed_config"):
        try:
            
            logger.info("  🔧 Registering SeqorthMoE with DeepSpeed...")
            
            # Get detected MoE class names from model (if available)
            # Fallback to default if not found
            moe_classes_to_add = [
                "SeqorthMoE",            # Top-level MoE wrapper
                "MixtralSparseMoeBlock", "Qwen2MoeSparseMoeBlock", "DeepseekV3MoE",
                # [Fix] Explicitly add Vision/Projector modules as leaves for ZeRO-3
                "SiglipVisionModel", "SiglipVisionTransformer", "SiglipEncoder",
                "SeqorthMultiModalProjector",
                # [Fix] Add Core Layers as leaves to ensure bias gathering
                "SeqorthAttention", "SeqorthMLP",
                # [TP Fix] Add Vision model classes as leaves to exclude from TP (prevents Conv3d weight dimension error)
                "Qwen3VLMoeVisionModel", "Qwen3VLMoeVisionBlock", "Qwen3VLMoeVisionPatchEmbed",
                "Qwen3VLMoeVisionPatchMerger", "Qwen3VLMoeVisionMLP", "Qwen3VLMoeVisionAttention",
                "Qwen3VLMoeTextExperts"
            ]
            if 'detected_moe_classes' in locals() and detected_moe_classes:
                for cls in detected_moe_classes:
                    if cls not in moe_classes_to_add:
                        moe_classes_to_add.append(cls)
                logger.info(f"  📋 Using detected MoE classes from model: {detected_moe_classes}")
            else:
                # Try to detect from model again (with PEFT wrapper support)
                moe_class_names_found = set()
                actual_model_for_detection = model
                if hasattr(model, 'base_model'):
                    actual_model_for_detection = model.base_model.model if hasattr(model.base_model, 'model') else model.base_model
                
                for name, module in actual_model_for_detection.named_modules():
                    # Check PEFT wrapper
                    if ModulesToSaveWrapper is not None and isinstance(module, ModulesToSaveWrapper):
                        if hasattr(module, 'modules_to_save'):
                            for adapter_name, inner_module in module.modules_to_save.items():
                                if 'SeqorthMoE' in type(inner_module).__name__:
                                    if hasattr(inner_module, 'experts'):
                                        for expert in inner_module.experts:
                                            moe_class_names_found.add(type(expert).__name__)
                    # Direct check
                    elif 'SeqorthMoE' in type(module).__name__:
                        # We don't want the dispatcher as a leaf, but we might want its experts
                        if hasattr(module, 'experts'):
                            for expert in module.experts:
                                moe_class_names_found.add(type(expert).__name__)
                
                if moe_class_names_found:
                    for cls in moe_class_names_found:
                        if cls not in moe_classes_to_add:
                            moe_classes_to_add.append(cls)
                    logger.info(f"  📋 Detected MoE classes from model: {moe_class_names_found}")
            
            # Detect Vision tower class from model (with PEFT wrapper support)
            vision_class_names_found = set()
            actual_model_for_vision = model
            if hasattr(model, 'base_model'):
                actual_model_for_vision = model.base_model.model if hasattr(model.base_model, 'model') else model.base_model
            
            for name, module in actual_model_for_vision.named_modules():
                # Check PEFT wrapper
                if ModulesToSaveWrapper is not None and isinstance(module, ModulesToSaveWrapper):
                    if hasattr(module, 'modules_to_save'):
                        for adapter_name, inner_module in module.modules_to_save.items():
                            module_class_name = type(inner_module).__name__
                            if any(v_cls in module_class_name for v_cls in ['Vision', 'Siglip', 'CLIP']):
                                if 'Model' in module_class_name or 'Transformer' in module_class_name:
                                    vision_class_names_found.add(module_class_name)
                # Direct check
                else:
                    module_class_name = type(module).__name__
                    if any(v_cls in module_class_name for v_cls in ['Vision', 'Siglip', 'CLIP']):
                        if 'Model' in module_class_name or 'Transformer' in module_class_name:
                            vision_class_names_found.add(module_class_name)
            if vision_class_names_found:
                for cls in vision_class_names_found:
                    if cls not in moe_classes_to_add:
                        moe_classes_to_add.append(cls)
                logger.info(f"  📋 Detected Vision classes from model: {vision_class_names_found}")
            
            # Method: Add moe_leaf_modules to DeepSpeed config file
            ds_cfg_path = model_config.get("deepspeed_config")
            if ds_cfg_path:
                # Use global json import (already imported at top of file)
                ds_cfg_path_abs = os.path.abspath(ds_cfg_path)
                
                # Read current config with error handling
                try:
                    with open(ds_cfg_path_abs, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse JSON with detailed error reporting
                    try:
                        ds_cfg = json.loads(content)
                    except json.JSONDecodeError as e:
                        # 상세한 오류 정보 제공
                        error_msg = f"Invalid JSON in DeepSpeed config file {ds_cfg_path_abs}\n"
                        error_msg += f"  Error: {e}\n"
                        error_msg += f"  File size: {len(content)} bytes\n"
                        
                        # 문제가 있는 줄 주변 표시
                        if hasattr(e, 'lineno') and e.lineno:
                            lines = content.split('\n')
                            error_line_num = e.lineno - 1  # 0-based index
                            error_msg += f"  Problem at line {e.lineno}, column {getattr(e, 'colno', '?')}:\n"
                            
                            # 주변 3줄 표시
                            start_line = max(0, error_line_num - 2)
                            end_line = min(len(lines), error_line_num + 3)
                            
                            for i in range(start_line, end_line):
                                line_num = i + 1
                                prefix = ">>> " if i == error_line_num else "    "
                                error_msg += f"  {prefix}Line {line_num}: {lines[i]}\n"
                        
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                except Exception as read_e:
                    logger.error(f"❌ Failed to read DeepSpeed config file: {read_e}")
                    raise
                
                # Add leaf_module configuration (DeepSpeed official format: zero_optimization.leaf_module)
                # Also support legacy moe_leaf_modules format for backward compatibility
                zero_opt = ds_cfg.get("zero_optimization", {})
                
                # Check for official format: zero_optimization.leaf_module
                leaf_module_cfg = zero_opt.get("leaf_module", {})
                if isinstance(leaf_module_cfg, dict):
                    existing_classes = leaf_module_cfg.get("classes", [])
                    if not isinstance(existing_classes, list):
                        existing_classes = list(existing_classes) if existing_classes else []
                else:
                    existing_classes = []
                
                # Also check legacy format: moe_leaf_modules (top-level)
                if "moe_leaf_modules" in ds_cfg:
                    legacy_classes = ds_cfg.get("moe_leaf_modules", [])
                    if isinstance(legacy_classes, list):
                        # Merge with existing classes from leaf_module
                        for cls in legacy_classes:
                            if cls not in existing_classes:
                                existing_classes.append(cls)
                
                # Add any missing MoE class names
                added = []
                for moe_class in moe_classes_to_add:
                    if moe_class not in existing_classes:
                        existing_classes.append(moe_class)
                        added.append(moe_class)
                
                # Update zero_optimization.leaf_module (official format)
                if "leaf_module" not in zero_opt:
                    zero_opt["leaf_module"] = {}
                zero_opt["leaf_module"]["classes"] = existing_classes
                if "names" not in zero_opt["leaf_module"]:
                    zero_opt["leaf_module"]["names"] = []
                if "name_suffixes" not in zero_opt["leaf_module"]:
                    zero_opt["leaf_module"]["name_suffixes"] = []
                
                ds_cfg["zero_optimization"] = zero_opt
                
                # Remove legacy moe_leaf_modules if present (migrated to leaf_module)
                if "moe_leaf_modules" in ds_cfg:
                    del ds_cfg["moe_leaf_modules"]
                    logger.info("  ℹ️  Migrated moe_leaf_modules to zero_optimization.leaf_module (official format)")
                
                if added:
                    logger.info(f"  ✅ Added MoE classes to leaf_module.classes: {added}")
                else:
                    logger.info(f"  ℹ️  All MoE classes already in leaf_module.classes: {moe_classes_to_add}")
                
                # Write back to config file with validation
                try:
                    # 임시 파일에 먼저 쓰기 (원자적 쓰기)
                    
                    temp_file = ds_cfg_path_abs + '.tmp'
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(ds_cfg, f, indent=4, ensure_ascii=False)
                    
                    # 쓰기 후 검증
                    with open(temp_file, 'r', encoding='utf-8') as f:
                        json.load(f)  # JSON 유효성 검증
                    
                    # 검증 통과 후 원본 파일로 교체
                    shutil.move(temp_file, ds_cfg_path_abs)
                    logger.info(f"  ✅ Updated DeepSpeed config: {ds_cfg_path_abs}")
                    logger.info(f"     moe_leaf_modules: {ds_cfg.get('moe_leaf_modules', [])}")
                except Exception as write_e:
                    # 임시 파일 정리
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    logger.error(f"❌ Failed to write DeepSpeed config file: {write_e}")
                    raise
            
            # CRITICAL: Also patch Accelerate's DeepSpeed plugin directly
            # This ensures Accelerate can find the MoE classes even if config file method fails
            # Strict Patching: No Fallbacks
            
            logger.info("  🔧 Patching Accelerate DeepSpeed plugin to handle missing MoE classes...")
            
            # First, try to find actual MoE classes in the model (with PEFT wrapper support)
            actual_moe_classes = set()
            actual_model_for_moe = model
            if hasattr(model, 'base_model'):
                actual_model_for_moe = model.base_model.model if hasattr(model.base_model, 'model') else model.base_model
            
            for name, module in actual_model_for_moe.named_modules():
                detected_module = None
                detected_class_name = None
                
                # Check PEFT wrapper
                if ModulesToSaveWrapper is not None and isinstance(module, ModulesToSaveWrapper):
                    if hasattr(module, 'modules_to_save'):
                        for adapter_name, inner_module in module.modules_to_save.items():
                            if hasattr(inner_module, 'experts') and (hasattr(inner_module, 'router') or hasattr(inner_module, 'gate')):
                                if "SeqorthMoE" in type(inner_module).__name__:
                                    continue
                                detected_module = inner_module
                                detected_class_name = type(inner_module).__name__
                                logger.debug(f"  Found PEFT-wrapped MoE module: {name} ({detected_class_name}, adapter: {adapter_name})")
                                break
                
                # Direct check for MoE-like modules (have experts and router/gate)
                if detected_module is None and hasattr(module, 'experts') and (hasattr(module, 'router') or hasattr(module, 'gate')):
                    # CRITICAL: Do NOT add SeqorthMoE (top-level wrapper) as a leaf module
                    if "SeqorthMoE" in type(module).__name__:
                        continue
                    detected_module = module
                    detected_class_name = type(module).__name__
                    logger.debug(f"  Found MoE module: {name} ({detected_class_name})")
                
                if detected_class_name:
                    actual_moe_classes.add(detected_class_name)
            
            # Add detected classes to moe_classes_to_add
            if actual_moe_classes:
                for cls_name in actual_moe_classes:
                    if cls_name not in moe_classes_to_add:
                        moe_classes_to_add.append(cls_name)
                logger.info(f"  📋 Found actual MoE classes in model: {actual_moe_classes}")
            
            # Update DeepSpeed config with all detected classes
            logger.info("  🚀 Standard DeepSpeed initialization (manual patches bypassed)")
                    
        except Exception as e:
            logger.warning(f"  ⚠️  DeepSpeed pre-init cleanup failed: {e}")
    
    # 데이터셋 검증
    train_dataset = dataset.get("train", None)
    eval_dataset = dataset.get("test", None)
    if eval_dataset is None:
        splited = train_dataset.train_test_split(test_size=0.1)
        train_dataset = splited["train"]
        eval_dataset = splited["test"]
    
    if train_dataset is None or len(train_dataset) == 0:
        raise ValueError(f"훈련 데이터셋이 비어있습니다! 데이터셋 로딩을 확인하세요.")
    
    logger.info(f"✅ Dataset validation: train={len(train_dataset)}, eval={len(eval_dataset) if eval_dataset else 0}")
    logger.debug(f"First sample keys: {list(train_dataset[0].keys())}")
    
    # CRITICAL: Ensure Accelerate DeepSpeed plugin is patched BEFORE Trainer initialization
    # This must happen right before Trainer creation to ensure the patch is active
    if model_config.get("deepspeed_config"):
        # Strict Patching: Ensure Accelerate DeepSpeed plugin is patched
        
        # Completely bypass set_moe_leaf_modules to avoid "Could not find" errors
        def noop_set_moe_leaf_modules(self, model):
            """No-op version that completely bypasses MoE class checking"""
            logger.debug("  ⚠️  Bypassed set_moe_leaf_modules check (custom MoE classes)")
            return
        
        logger.info("  🔧 Forcing idempotent patching of DeepSpeedPlugin...")
        # Direct intervention if needed
        acc_dataclasses.DeepSpeedPlugin.set_moe_leaf_modules = noop_set_moe_leaf_modules
        logger.info("  ✅ Applied final patch to Accelerate DeepSpeedPlugin.set_moe_leaf_modules (no-op)")

        # CRITICAL: Manually enforce ZeRO-3 leaf modules on the model instance
        # This bypasses config issues and ensures granular partitioning at Layer level
        try:
            leaf_classes = set()
            
            # 1. Identify Decoder Layer Class (Recursive search)
            layers = None
            if hasattr(model, 'layers'):
                layers = model.layers
            elif hasattr(model, 'model'):
                if hasattr(model.model, 'layers'):
                    layers = model.model.layers
                elif hasattr(model.model, 'model'):
                    if hasattr(model.model.model, 'layers'):
                        layers = model.model.model.layers
            
            if layers is None and hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
                layers = model.language_model.layers
            
            if layers is not None and len(layers) > 0:
                decoder_cls = type(layers[0])
                leaf_classes.add(decoder_cls)
                logger.info(f"  🍃 Identified Decoder Layer Class: {decoder_cls.__name__}")
            else:
                 logger.warning(f"  ⚠️  Could not find 'layers' for Leaf Module detection! Model keys: {list(model.__dict__.keys()) if hasattr(model, '__dict__') else 'N/A'}")
            
            # 2. Explicitly add a minimal leaf-module set for ZeRO-3 stability.
            # Keep this list tight to avoid broad all-gather synchronization points.
            important_classes = [
                # Qwen3-VL-MoE Text (do NOT mark Qwen3VLMoeTextExperts as leaf)
                "Qwen3VLMoeTextSparseMoeBlock"
            ]
            for name, module in model.named_modules():
                class_name = type(module).__name__
                if class_name in important_classes:
                    leaf_classes.add(type(module))
            
            # 3. Apply to DeepSpeed utils
            if leaf_classes:
                # Remove any None or invalid items just in case
                leaf_classes = {c for c in leaf_classes if c is not None}
                deepspeed.utils.set_z3_leaf_modules(model, list(leaf_classes))
                logger.info(f"  ✅ Forced ZeRO-3 leaf modules: {[c.__name__ for c in leaf_classes]}")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to manually set ZeRO-3 leaf modules: {e}")

    # Ensure gradient checkpointing and input grads are set correctly before Trainer
    if training_config.get("gradient_checkpointing", False):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            logger.info("✅ Enabled input require grads for gradient checkpointing (pre-trainer)")
        else:
            logger.warning("⚠️ Model does not have enable_input_require_grads method")
        # [0vs2048] Vision encoder는 rank별 patch 수(256 vs 1064)가 달라 checkpoint 재계산 시 shape 불일치 → vision만 checkpoint 비활성화
        def _disable_vision_checkpoint(m):
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = False
        for candidate in (model, getattr(model, "model", None), getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
            if candidate is None:
                continue
            vision_root = getattr(candidate, "visual", None)
            if vision_root is not None:
                break
        else:
            vision_root = None
        if vision_root is not None:
            for m in vision_root.modules():
                _disable_vision_checkpoint(m)
            logger.warning("[0vs2048] Vision encoder gradient_checkpointing disabled (rank별 patch 수 불일치 방지)")
    
    # CRITICAL: DeepSpeed autotp will handle embedding layers automatically
    # We don't need to verify embedding shapes as autotp may have already sharded them
    # or will handle them during engine initialization
    logger.info("  ℹ️  Skipping embedding layer verification - DeepSpeed autotp will handle embeddings")
    


    # ZeRO-2 CUDA OOM fix: force offload_optimizer (nvme) into the exact config file
    # that Accelerate/DeepSpeed will read at ds_initialize so fp32 partition goes to CPU (not 61GB on GPU)
    _ds_path = model_config.get("deepspeed_config")
    if _ds_path and isinstance(_ds_path, str) and os.path.exists(_ds_path):
        try:
            with open(os.path.abspath(_ds_path), "r", encoding="utf-8") as f:
                _ds_cfg = json.load(f)
            _zo = _ds_cfg.get("zero_optimization") or {}
            if _zo.get("stage") == 2:
                _off = _zo.get("offload_optimizer") or {}
                _zo["offload_optimizer"] = {
                    "device": "nvme",
                    "nvme_path": _off.get("nvme_path") or "/mls/conan/offload",
                    "pin_memory": False,
                    "fast_init": True,
                    "ratio": 1,
                    "buffer_count": 8,
                    "pipeline_read": False,
                    "pipeline_write": False,
                }
                _ds_cfg["zero_optimization"] = _zo
                if dist.is_initialized() and is_main_process():
                    with open(os.path.abspath(_ds_path), "w", encoding="utf-8") as f:
                        json.dump(_ds_cfg, f, indent=2, ensure_ascii=False)
                    logger.info("  ✅ [ZeRO-2] Forced offload_optimizer nvme into DS config file before Trainer (avoids fp32 GPU OOM)")
                if dist.is_initialized():
                    dist.barrier()
        except Exception as _e:
            logger.warning(f"  ⚠️ Could not enforce offload_optimizer in DS config: {_e}")

    # CRITICAL: SymmetricSFTTrainer for ZeRO-3 + Multimodal MoE
    # Use standard DistributedSampler - BalancedVisionSampler caused severe bottleneck
    # by iterating through entire dataset (2M+ samples) during initialization
    class SymmetricSFTTrainer(SFTTrainer):
        def _get_train_sampler(self, dataset=None) -> torch.utils.data.Sampler:
            target_dataset = dataset if dataset is not None else self.train_dataset
            if target_dataset is None:
                return None
            
            # Use standard DistributedSampler to avoid initialization bottleneck
            return torch.utils.data.distributed.DistributedSampler(
                target_dataset,
                num_replicas=dist.get_world_size() if dist.is_available() else 1,
                rank=dist.get_rank() if dist.is_available() else 0,
                shuffle=True,
                seed=self.args.seed
            )
        
        def _get_eval_sampler(self, eval_dataset) -> torch.utils.data.Sampler:
            if eval_dataset is None:
                return None
            return torch.utils.data.distributed.DistributedSampler(
                eval_dataset,
                num_replicas=dist.get_world_size() if dist.is_available() else 1,
                rank=dist.get_rank() if dist.is_available() else 0,
                seed=self.args.seed
            )

    trainer = SymmetricSFTTrainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collate_fn,
        optimizers=(optimizer_for_trainer, None) if optimizer_for_trainer is not None else (None, None)
    )
    
    # RPG disabled under ZeRO-3: p.sum()*0.0 does not trigger DS gradient hooks for partitioned params.
    # Instead, patch unscale_and_clip_grads to treat None grad as zero (correct for unused MoE experts).
    # _patch_trainer_for_participation_guard(trainer)
    
    # CRITICAL: Verify and fix embedding layers after Trainer creation but before DeepSpeed engine init
    # DeepSpeed autotp may incorrectly shard embedding weights during engine initialization
    try:
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model = actual_model.base_model if hasattr(actual_model, 'base_model') else actual_model
        
        def verify_and_fix_embeddings(model_to_check, prefix=""):
            """Recursively verify and fix embedding layers"""
            
            # Get config for vocab_size and hidden_size
            config = None
            if hasattr(model_to_check, 'config'):
                config = model_to_check.config
            elif hasattr(model_to_check, 'base_model') and hasattr(model_to_check.base_model, 'config'):
                config = model_to_check.base_model.config
            
            vocab_size = None
            hidden_size = None
            if config:
                vocab_size = getattr(config, 'vocab_size', None)
                hidden_size = getattr(config, 'hidden_size', None)
                if not hidden_size and hasattr(config, 'text_config'):
                    hidden_size = getattr(config.text_config, 'hidden_size', None)
            
            for name, module in model_to_check.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(module, nn.Embedding):
                    # Skip positional embeddings (can be 1-D)
                    if 'pos_embed' in full_name or 'positional' in full_name.lower():
                        continue
                    
                    if hasattr(module, 'weight') and module.weight is not None:
                        weight = module.weight
                        
                        # Check for empty embeddings
                        if weight.numel() == 0:
                            logger.warning(f"⚠️  Embedding {full_name} has empty weight")
                            if vocab_size and hidden_size:
                                logger.info(f"  🔧 Reinitializing embedding {full_name}")
                                new_weight = torch.empty(vocab_size, hidden_size, dtype=weight.dtype, device=weight.device)
                                nn.init.normal_(new_weight, mean=0.0, std=0.02)
                                module.weight = nn.Parameter(new_weight)
                                logger.info(f"  ✅ Fixed empty embedding {full_name}")
                            continue
                        
                        # Token embeddings should be 2-D
                        if weight.dim() != 2:
                            logger.warning(f"⚠️  Embedding {full_name} has dim {weight.dim()}, shape: {weight.shape}")
                            if weight.dim() == 1 and vocab_size and hidden_size and weight.numel() == vocab_size * hidden_size:
                                new_weight = weight.view(vocab_size, hidden_size)
                                module.weight.data = new_weight
                                logger.info(f"  ✅ Fixed embedding {full_name} shape: {weight.shape} -> {new_weight.shape}")
                else:
                    # Recursively check submodules
                    verify_and_fix_embeddings(module, full_name)
        
        verify_and_fix_embeddings(actual_model)
        logger.info("  ✅ Verified and fixed embedding layers after Trainer creation")
    except Exception as e:
        logger.warning(f"  ⚠️  Embedding verification failed (may be normal if model is wrapped): {e}")

    # CRITICAL: Freeze vision encoder/tower AFTER Trainer creation
    # This ensures DeepSpeed wrapping doesn't undo the freeze
    # This prevents TP/SP from affecting vision embeddings and simplifies training
    freeze_vision = model_config.get("freeze_vision", False)
    if freeze_vision:
        logger.info("=" * 80)
        logger.info("🔒 Freezing vision encoder/tower (after Trainer creation)")
        logger.info("=" * 80)
        
        vision_frozen_params = 0
        vision_modules_found = []
        
        # Get actual model (may be wrapped by Trainer/DeepSpeed)
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model = actual_model.base_model if hasattr(actual_model, 'base_model') else actual_model
        
        # Method 1: Use vision_tower property if available
        vision_tower = None
        try:
            if hasattr(actual_model, 'vision_tower'):
                vision_tower = actual_model.vision_tower
            elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'vision_tower'):
                vision_tower = actual_model.model.vision_tower
            elif hasattr(actual_model, 'visual'):
                vision_tower = actual_model.visual
        except:
            pass
        
        if vision_tower is not None:
            # Freeze entire vision tower
            for name, param in vision_tower.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    vision_frozen_params += param.numel()
            vision_modules_found.append("vision_tower")
            logger.info(f"  ✅ Frozen vision_tower via property: {format_parameters(vision_frozen_params)} parameters")
        else:
            # Method 2: Find vision modules by name and class
            vision_module_names = ['visual', 'vision_tower', 'vision_model', 'vision_encoder']
            for name, module in actual_model.named_modules():
                # Check if this is a vision module by name
                is_vision_module = False
                for vision_name in vision_module_names:
                    if vision_name in name.lower() or name.startswith(vision_name):
                        is_vision_module = True
                        break
                
                # Also check by class name
                if not is_vision_module:
                    class_name = type(module).__name__
                    if any(v in class_name for v in ['Vision', 'Siglip', 'CLIP', 'SeqorthVLVision', 'Seqorth2VLVision']):
                        is_vision_module = True
                
                if is_vision_module:
                    module_frozen = 0
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.requires_grad:
                            param.requires_grad = False
                            vision_frozen_params += param.numel()
                            module_frozen += param.numel()
                    if module_frozen > 0:
                        vision_modules_found.append(name)
            
            if vision_modules_found:
                logger.info(f"  ✅ Frozen {len(vision_modules_found)} vision modules: {vision_modules_found[:5]}{'...' if len(vision_modules_found) > 5 else ''}")
                logger.info(f"  ✅ Frozen {format_parameters(vision_frozen_params)} vision parameters")
            else:
                logger.warning("  ⚠️  freeze_vision=True but no vision modules found to freeze")
        
        if vision_frozen_params > 0:
            logger.info(f"  ✅ Vision encoder/tower frozen: {format_parameters(vision_frozen_params)} parameters")
            # Protect vision freeze from being undone
            protect_vision_freeze(model, logger)
        else:
            logger.warning("  ⚠️  No vision parameters were frozen")
    else:
        logger.info("  ℹ️  Vision encoder/tower is trainable (freeze_vision=False)")
    
    logger.info("=" * 80)
    
    # Trainer 생성 후 wandb가 초기화되었는지 확인하고, 필요시 초기화
    if training_config.get("report_to", None) and "wandb" in training_config["report_to"]:
        rank = int(os.getenv("RANK", "0"))
        if rank == 0 and (wandb.run is None or not wandb.run):
            # Trainer가 아직 wandb를 초기화하지 않았다면 여기서 초기화
            run = wandb.init(
                project="seqorth-sft",
                name=training_config["run_name"],
                config=config,
                mode="online"  # 항상 online으로 wandb에 기록
            )
            run.define_metric("train/*", step_metric="train/global_step")
            run.define_metric("validation/*", step_metric="validation/step")
            run.define_metric("eval/*", step_metric="eval/step")
            run.define_metric("moe/*", step_metric="train/global_step")
            run.define_metric("multi_modality/*", step_metric="train/global_step")
            run.define_metric("router/*", step_metric="train/global_step")
            run.define_metric("other/*", step_metric="train/global_step")

            logger.info("✅ wandb initialized after Trainer creation")
    
    
    # Add MoE monitoring callback
    trainer.add_callback(
        create_moe_callback_for_transformers(
            num_experts=model_config.get("seqorth_params", {}).get("n_routed_experts", 8),
            log_every_n_steps=1,
            logger=wandb,
            log_to_console=False,
            debug_logging=True,
            tokenizer=tokenizer,
            log_heatmap_every=5,             
            alert_threshold_imbalance=4.0,   
            unused_expert_threshold=0.25,    
            entropy_threshold=0.1,           
            save_detailed_logs=False,        
            enable_generation_logging=False,
        ))
    logger.info("✅ MoE monitoring callback added")
    
    # CRITICAL: PEFT modules_to_save 동기화 callback 추가
    modules_to_save_sync_callback = ModulesToSaveSyncCallback(sync_every_n_steps=10, logger=logger)
    trainer.add_callback(modules_to_save_sync_callback)
    logger.info("✅ ModulesToSaveSyncCallback added")
    
    # 배치 정보를 저장하는 callback (OOM 디버깅용)
    batch_tracker = BatchTrackingCallback(trainer)
    trainer.add_callback(batch_tracker)

    # ===== Benchmark evaluation callback =====
    benchmark_eval_enabled = training_config.get("enable_benchmark_eval", True)
    benchmark_eval_tasks = training_config.get(
        "benchmark_eval_tasks",
        ['mmlu', 'hellaswag', 'gsm8k', 'truthfulqa', 'arc', 'ifeval', 'mme', 'vqav2', 'textvqa'],
    )
    benchmark_eval_mode = training_config.get("benchmark_eval_mode", "step")
    if benchmark_eval_mode not in {"step", "epoch"}:
        benchmark_eval_mode = "step"

    default_benchmark_freq = 1000
    benchmark_eval_frequency = int(training_config.get("benchmark_eval_frequency", default_benchmark_freq) or default_benchmark_freq)

    if benchmark_eval_enabled:
        logger.info(
            f"✅ Enabling benchmark callback (mode={benchmark_eval_mode}, freq={benchmark_eval_frequency}, tasks={benchmark_eval_tasks})"
        )
        trainer.add_callback(
            ModelEvalCallback(
                trainer=trainer,
                enable_benchmarks=True,
                benchmarks_to_run=benchmark_eval_tasks,
                benchmark_eval_frequency=benchmark_eval_frequency,
                eval_mode=benchmark_eval_mode,
                mme_max_samples=training_config.get("benchmark_mme_max_samples", 5),
                benchmark_max_samples_per_task=training_config.get("benchmark_max_samples_per_task", 3),
                benchmark_gsm8k_max_samples=training_config.get("benchmark_gsm8k_max_samples", 3),
                benchmark_max_tasks=training_config.get("benchmark_max_tasks"),
                benchmark_max_new_tokens=training_config.get("benchmark_max_new_tokens", 64),
                benchmark_disable_cot=training_config.get("benchmark_disable_cot", True),
                benchmark_ifeval_max_samples=training_config.get("benchmark_ifeval_max_samples", 5),
            )
        )
    else:
        logger.info("ℹ️ Benchmark callback disabled (enable_benchmark_eval=False)")

    logger.info("✅ All callbacks restored")

    # Print training info
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    if model_config.get("initialize_from_scratch", False):
        print(f"Model: Initializing from scratch (small model)")
    else:
        print(f"Model: {model_config.get('model_name_or_path', 'N/A')}")
    
    # Calculate and print model parameters
    try:
        # Get actual model (handle DeepSpeed wrapping)
        model_to_count = trainer.model
        if hasattr(model_to_count, 'module'):
            model_to_count = model_to_count.module
        
        # Count total parameters (using unique counting to avoid shared module duplication)
        total_params = count_unique_parameters(model_to_count, verbose=False)
        trainable_params = sum(p.numel() for p in model_to_count.parameters() if p.requires_grad)
        
        # Format in B (billion) or M (million) units
        if total_params >= 1e9:
            total_str = f"{total_params / 1e9:.2f}B"
        elif total_params >= 1e6:
            total_str = f"{total_params / 1e6:.2f}M"
        else:
            total_str = f"{total_params / 1e3:.2f}K"
        
        if trainable_params >= 1e9:
            trainable_str = f"{trainable_params / 1e9:.2f}B"
        elif trainable_params >= 1e6:
            trainable_str = f"{trainable_params / 1e6:.2f}M"
        else:
            trainable_str = f"{trainable_params / 1e3:.2f}K"
        
        print(f"Model parameters: {total_str} total, {trainable_str} trainable")
    except Exception as e:
        print(f"Model parameters: Unable to calculate ({e})")
    
    print(f"Dataset: {data_config['dataset_name']}")
    print(f"Max sequence length: {data_config['max_seq_length']}")
    print(f"Use LoRA: {model_config['use_lora']}")
    if model_config['use_lora']:
        print(f"LoRA rank: {model_config['lora_r']}")
    print(f"DeepSpeed config: {model_config.get('deepspeed_config', 'None')}")
    print(f"Training epochs: {training_config['num_train_epochs']}")
    print(f"Batch size per device: {training_config['per_device_train_batch_size']}")
    print(f"Gradient accumulation steps: {training_config['gradient_accumulation_steps']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"FP16: {training_config['fp16']}")
    print(f"BF16: {training_config['bf16']}")
    print("="*50)
    # summary(
    #     trainer.model,
    #     input_data={
    #         'input_ids': torch.randint(0, tokenizer.tokenizer.vocab_size, (1, 1024), device=trainer.model.device)
    #     }, depth=3)
    # Start training
    print("Starting training...")
    # Guard heavy profiler behind an env flag to avoid OOM from profiler buffers during full training
    try:
        # Log training start
        logger.info(f"🚀 Starting training...")
        logger.info(f"🔧 Training configuration:")
        logger.info(f"  - Epochs: {training_config['num_train_epochs']}")
        logger.info(f"  - Batch size per device: {training_config['per_device_train_batch_size']}")
        logger.info(f"  - Gradient accumulation steps: {training_config['gradient_accumulation_steps']}")
        logger.info(f"  - Learning rate: {training_config['learning_rate']}")
        logger.info(f"  - Max sequence length: {data_config['max_seq_length']}")
        
        # eval 최적화를 위한 커스텀 eval 함수 설정
        logger.info("🔧 Setting up memory-optimized evaluation...")
        original_eval_fn = getattr(trainer, 'evaluate', None)
        trainer.evaluate = lambda eval_dataset=None, ignore_keys=None, metric_key_prefix="eval": eval_with_memory_optimization(
            trainer,
            original_eval_fn,
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            logger=logger)
        
        # 학습 시작 전 메모리 정리
        logger.info("🧹 학습 시작 전 GPU 메모리 정리...")
        
        # Set allocation config to reduce fragmentation
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        clear_gpu_memory(logger)
        
        # DataLoader 최적화 (bottleneck 방지)
        # CRITICAL: num_workers=0으로 설정하여 메인 프로세스에서 직접 처리
        # Vision 샘플 처리 시 multiprocessing이 bottleneck을 유발할 수 있음
        if hasattr(trainer.args, 'dataloader_num_workers'):
            if trainer.args.dataloader_num_workers is None or trainer.args.dataloader_num_workers > 0:
                logger.info(f"🔧 DataLoader num_workers를 0으로 설정 (vision processing bottleneck 방지)")
                trainer.args.dataloader_num_workers = 0
        
        # Log initial memory state
        log_gpu_memory(logger, "TRAINING_START")
        
        # CRITICAL FIX: Force entire sequence learning by patching compute_loss
        original_compute_loss = trainer.compute_loss
        _compute_loss_step = [0]  # first step batch log

        def patched_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            # [0vs2048] Participation Guard (v16) - Global Connection
            global _PARTICIPATION_POOL
            if '_PARTICIPATION_POOL' in globals():
                _PARTICIPATION_POOL.clear()

            # [0vs2048] 첫 스텝에서만 배치 vision 정보 로그 (모든 rank, 체크용)
            step = _compute_loss_step[0]
            _compute_loss_step[0] += 1
            if step == 0 and os.environ.get("SEQORTH_DEBUG_0VS2048") == "1":
                try:
                    r = dist.get_rank() if dist.is_initialized() else 0
                    kv = [f"rank={r}"]
                    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                        p = inputs["pixel_values"]
                        kv.append(f"pixel_values={tuple(p.shape) if hasattr(p, 'shape') else type(p).__name__}")
                        num_patches = int(p.shape[0]) if hasattr(p, "shape") and len(p.shape) > 0 else None
                        kv.append(f"num_patches={num_patches}")
                    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
                        g = inputs["image_grid_thw"]
                        kv.append(f"image_grid_thw={tuple(g.shape) if hasattr(g, 'shape') else type(g).__name__}")
                    if "input_ids" in inputs:
                        kv.append(f"input_ids={tuple(inputs['input_ids'].shape)}")
                    # [0vs2048] num_image_tokens_in_input_ids: if any rank has 0, that rank gets 0-sized grad to vision → 0 vs 2048
                    num_image_tokens_local = 0
                    cfg = getattr(getattr(model, "module", model), "config", None)
                    image_token_id = getattr(cfg, "image_token_id", getattr(cfg, "image_token_index", None)) if cfg else None
                    if "input_ids" in inputs and image_token_id is not None:
                        num_image_tokens_local = (inputs["input_ids"] == image_token_id).sum().item()
                    kv.append(f"num_image_tokens_in_input_ids={num_image_tokens_local}")
                    if dist.is_initialized():
                        dev = inputs["input_ids"].device if "input_ids" in inputs else None
                        if dev is not None:
                            t = torch.tensor([num_image_tokens_local], dtype=torch.long, device=dev)
                            dist.all_reduce(t, op=dist.ReduceOp.MIN)
                            min_image_tokens = t.item()
                            if min_image_tokens == 0:
                                logger.warning(
                                    "[0vs2048] At least one rank has num_image_tokens_in_input_ids=0 → backward produces 0-sized gradient for vision → 0 vs 2048. Fix data/collator so every rank has image tokens in input_ids."
                                )
                    logger.warning(f"[0vs2048] STEP0 BATCH CHECK: {', '.join(kv)}")
                    if r == 0:
                        logger.warning("[0vs2048] STEP0 CHECK: Compare num_patches and num_image_tokens_in_input_ids across ranks; if any rank has 0 image tokens → 0vs2048 (see traceback).")
                except Exception as _e:
                    logger.warning(f"[0vs2048] STEP0 batch log failed: {_e}")
            
            # EXPLICIT VISION EXECUTION (v20): Force Vision Tower participation
            # If no images are present, we MUST manually run the vision tower and add it to loss.
            # Merely injecting pixel_values into inputs is insufficient because qwen3 filters them
            # based on input_ids tokens. We bypass the filter by running .visual directly.
            vision_guard = 0.0
            if inputs.get("pixel_values") is None or (torch.is_tensor(inputs["pixel_values"]) and inputs["pixel_values"].numel() == 0):
                # Check device/dtype from input_ids
                ref_tensor = inputs["input_ids"]
                device = ref_tensor.device
                dtype = torch.bfloat16
                
                # Create dummy inputs
                dummy_pixels = torch.zeros((1, 1152), device=device, dtype=dtype, requires_grad=True)
                dummy_grid = torch.tensor([[1, 1, 1]], device=device, dtype=torch.int32)
                
                # Find the vision tower (it could be deeply nested in DeepSpeed/Accelerate wrappers)
                vision_module = None
                if hasattr(model, "visual"):
                    vision_module = model.visual
                elif hasattr(model, "model") and hasattr(model.model, "visual"):
                    vision_module = model.model.visual # Standard Qwen3VLMoeForConditionalGeneration
                elif hasattr(model, "module"): # DeepSpeed/DDP wrapper
                    if hasattr(model.module, "visual"):
                        vision_module = model.module.visual
                    elif hasattr(model.module, "model") and hasattr(model.module.model, "visual"):
                        vision_module = model.module.model.visual

                if vision_module:
                    # Run vision tower explicitly
                    # Qwen3 vision forward: (hidden_states, grid_thw)
                    try:
                        # We use 0.0 * output.sum() to attach to graph without affecting loss value
                        # Note: Qwen3 vision returns (hidden_states, deepstack_features) usually
                        v_out = vision_module(dummy_pixels, dummy_grid)
                        if isinstance(v_out, tuple):
                            vision_guard = v_out[0].sum() * 0.0
                        else:
                            vision_guard = v_out.sum() * 0.0
                    except Exception as ve:
                        logger.warning(f"[0vs2048] Vision guard execution failed: {ve}")
            
            # Execute original compute_loss with Logits Guard (v18)
            # We must ensure the backbone (LM Head/Embeddings) is connected
            # even if all labels are -100 (which prunes the grad path).
            
            # Re-implementing the call to ensure we catch the output
            if return_outputs:
                loss, outputs = original_compute_loss(model, inputs, return_outputs=True)
            else:
                # To get logits, we need outputs. We call with return_outputs=True internally.
                try:
                    loss, outputs = original_compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
                except TypeError:
                    loss, outputs = original_compute_loss(model, inputs, return_outputs=True)

            # POOL GUARD (v18): Ensure every skipped module participates
            guard = 0.0
            if '_PARTICIPATION_POOL' in globals():
                pool = globals()['_PARTICIPATION_POOL']
                while pool:
                    guard = guard + pool.pop()
            
            # LOGITS GUARD: Connect logits to ensure backbone participation
            if hasattr(outputs, "logits") and outputs.logits is not None:
                # Add tiny grad path to logits
                guard = guard + (outputs.logits.sum() * 0.0)
            
            # Add Vision Guard
            guard = guard + vision_guard
            
            if return_outputs:
                return (loss + guard, outputs)
            return loss + guard

        trainer.compute_loss = patched_compute_loss

        # CRITICAL: Apply DeepSpeed gradient patch BEFORE training starts
        # DeepSpeed is initialized when trainer.train() is called, so we need to patch right before that
        # [ZeRO-3 Fix] REMOVED Gradient Patch.
        # This patch forces param.grad initialization for frozen parameters, which creates
        # partitioned (size 0) gradients that conflict with full-sized gradients during backward
        # in ZeRO-3 leaf modules. DeepSpeed handles None grads correctly, so this is unnecessary.
        logger.info("🔧 DeepSpeed gradient patch skipped (causes shape mismatch in ZeRO-3 leaf modules)")
        
        # CRITICAL FIX: Disable checkpoint debug mode to prevent hanging
        # Debug mode can cause issues with DeepSpeed ZeRO-3 and distributed training
        logger.info("🔍 Configuring gradient checkpointing (debug mode disabled for stability)...")
        # Only enable debug mode if explicitly requested via environment variable
        checkpoint_debug = bool(int(os.getenv("CHECKPOINT_DEBUG", "0")))
        if checkpoint_debug:
            torch.utils.checkpoint.set_checkpoint_debug_enabled(True)
            logger.warning("⚠️ Checkpoint debug mode enabled - may cause hanging in distributed training")
        else:
            torch.utils.checkpoint.set_checkpoint_debug_enabled(False)
        
        # Start training with progress monitoring
        start_time = time.time()
        
        # 학습 시작 전 상태 출력
        print("\n" + "="*80)
        print("🚀 STARTING TRAINING")
        print("="*80)
        sys.stdout.flush()
        logger.info("🚀 Starting training...")
        
        if not (dist.is_available() and dist.is_initialized()):
            print(f"⚠️ Rank {os.environ.get('RANK', '0')}: Dist not initialized! Calling trainer.accelerator.state to check...")
            try:
                # Accessing accelerator state might trigger init if lazily created
                _ = trainer.accelerator.state
            except:
                pass

        if dist.is_available() and dist.is_initialized():
             logger.info(f"✅ Rank {dist.get_rank()}: Dist initialized. World size: {dist.get_world_size()}")
             dist.barrier()
        else:
             print("❌ CRITICAL: Dist still not initialized after Trainer check!")
        
        # CRITICAL: Verify model consistency across ranks before training
        # This prevents "Detected mismatch between collectives on ranks" at step 8
        print(f"🔍 Rank {os.environ.get('RANK', '0')}: Running pre-training consistency check...")
        sys.stdout.flush()
        logger.info("🔍 Running pre-training consistency check...")
        verify_model_consistency(trainer.model, logger, output_dir=training_args.output_dir)
        

        
        enable_profiler = bool(int(os.getenv("PROFILE_TRAINING", "0")))
        
        # [COMPONENT TEST MODE] 첫 번째 스텝 후 종료하는 콜백 추가
        _test_mode = os.environ.get("SEQORTH_TEST_MODE", "0") == "1"
        _test_max_steps = int(os.environ.get("SEQORTH_TEST_MAX_STEPS", "1"))
        
        if _test_mode:
            from transformers import TrainerCallback
            
            class EarlyStopCallback(TrainerCallback):
                def __init__(self, max_steps=1):
                    self.max_steps = max_steps
                    
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step >= self.max_steps:
                        print(f"\n[TEST MODE] ✅ Step {state.global_step} completed successfully!")
                        print(f"[TEST MODE] Stopping training after {self.max_steps} step(s).")
                        control.should_training_stop = True
                    return control
            
            trainer.add_callback(EarlyStopCallback(max_steps=_test_max_steps))
            logger.info(f"[TEST MODE] Added EarlyStopCallback (max_steps={_test_max_steps})")
        
        # DETECT_ANOMALY is set earlier in setup; when ON, backward 실패 시 정확한 연산 위치 출력됨.

        try:
            # ZeRO-3: Sync all ranks before training to avoid collective mismatch / deadlock
            if dist.is_initialized():
                dist.barrier()
            if enable_profiler:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    trainer.train()
                    profiler_table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
                    wandb.log({"profiler_table": wandb.Table(data=[profiler_table])})
            else:
                # CRITICAL FIX: Don't wrap trainer.train() with checkpoint debug mode
                trainer.train()
            
            training_time = time.time() - start_time
            logger.info(f"✅ Training completed successfully in {training_time:.2f} seconds")

            sys.stdout.flush()
            sys.stderr.flush()
        except Exception as train_e:
            # trainer.train() 내부에서 발생한 오류를 즉시 출력
            _erank = dist.get_rank() if dist.is_initialized() else -1
            print("\n" + "="*80)
            print("🚨 TRAINING ERROR DETECTED INSIDE trainer.train()")
            print("="*80)
            print(f"Error type: {type(train_e).__name__}")
            print(f"Error message: {str(train_e)}")
            print(f"[0vs2048] rank that raised: {_erank}")
            print("[0vs2048] SUMMARY: If STEP0 BATCH CHECK showed same num_patches on all ranks, 0vs2048 is NOT from vision patch mismatch → see Full traceback for exact op (AccumulateGrad).")
            print(f"\nFull traceback:")
            print(traceback.format_exc())
            print("="*80)
            sys.stdout.flush()
            sys.stderr.flush()
            
            logger.error(f"❌ Training error inside trainer.train(): {type(train_e).__name__}: {str(train_e)}")
            logger.error(f"❌ Traceback:\n{traceback.format_exc()}")
            
            # 오류를 다시 raise하여 상위 except 블록에서 처리
            raise train_e
        
    except torch.OutOfMemoryError as e:
        # CUDA OOM 전용 처리
        handle_cuda_oom(e, trainer, logger)
        raise e
        
    except MemoryError as e:
        # 로컬 RAM OOM 전용 처리
        handle_ram_oom(e, trainer, logger)
        raise e
        
    except KeyboardInterrupt as e:
        handle_training_exception(e, trainer, logger, context="training_keyboard_interrupt")
        raise e
        
    except RuntimeError as e:
        # DistBackendError (NCCL 오류 등)를 명시적으로 처리
        error_msg = str(e)
        error_type = type(e).__name__
        if "NCCL" in error_msg or "nccl" in error_msg.lower() or "DistBackendError" in error_type:
            print("\n" + "="*80)
            print("🚨 NCCL/DISTRIBUTED COMMUNICATION ERROR")
            print("="*80)
            print(f"Error type: {error_type}")
            print(f"Error message: {error_msg}")
            if "Watchdog" in error_msg and "timeout" in error_msg.lower():
                print("💡 Rank 1 timed out waiting for all_gather → Rank 0 likely exited earlier (e.g. CUDA OOM).")
                print("   Check Rank 0 stdout/stderr for OutOfMemoryError. ZeRO-3 only (no fallback).")
                print("   Reduce OOM: lower stage3_max_live_parameters in deepspeed_zero3.json, gradient_checkpointing=True, batch_size=1.")
            print("="*80)
            sys.stdout.flush()
            sys.stderr.flush()
        
        handle_training_exception(e, trainer, logger, context="training_runtime_error")
        raise e
        
    except Exception as e:
        handle_training_exception(e, trainer, logger, context="training")
        raise e
        
    finally:
        # 원래 eval 함수 복원
        # Save final model (실패해도 evaluation은 실행)
        sys.stdout.flush()
        sys.stderr.flush()
        
        # 오류가 발생했는지 확인
        if sys.exc_info()[0] is not None:
            exc_type, exc_value, exc_tb = sys.exc_info()
            print("\n" + "="*80)
            print("⚠️  FINALLY BLOCK: Exception detected during training")
            print("="*80)
            print(f"Exception type: {exc_type.__name__}")
            print(f"Exception value: {exc_value}")
            if exc_tb:
                print(f"\nTraceback:")
                print(''.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            print("="*80)
            sys.stdout.flush()
            sys.stderr.flush()
            
            logger.warning(f"⚠️  Exception detected in finally block: {exc_type.__name__}: {exc_value}")
        
        # Save final model (Strict: No Fallbacks)
        model_saved = False
        try:
            print("Saving final model...")
            logger.info("💾 Saving final model...")
            sys.stdout.flush()
            
            # Use DeepSpeed checkpoint save to avoid NVMe swap buffer conflict
            # When stage3_gather_16bit_weights_on_model_save=false, use save_checkpoint
            if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
                logger.info("💾 Using DeepSpeed save_checkpoint (NVMe-safe)...")
                checkpoint_dir = os.path.join(training_args.output_dir, "deepspeed_checkpoint")
                os.makedirs(checkpoint_dir, exist_ok=True)
                trainer.deepspeed.save_checkpoint(checkpoint_dir)
                logger.info(f"✅ DeepSpeed checkpoint saved to {checkpoint_dir}")
                logger.info("💡 Use zero_to_fp32.py to convert to fp32 weights")
                model_saved = True
            else:
                # Standard save for non-DeepSpeed
                trainer.save_model()
                logger.info("✅ Model saved")
                model_saved = True
            
        except Exception as e:
            logger.error(f"❌ Model save failed: {e}")
            logger.warning("⚠️ Training completed but model save failed. Continuing without save...")
            # Don't raise - training was successful, just save failed

        
        # Save tokenizer (실패해도 evaluation은 실행)
        try:
            tokenizer.save_pretrained(training_args.output_dir)
            logger.info("✅ Tokenizer saved")
        except Exception as tokenizer_e:
            logger.warning(f"⚠️ Tokenizer save failed: {tokenizer_e}")
        
        print("Training End")
        logger.info("🏁 Training End")
        
        if original_eval_fn:
            logger.debug("🔧 Restoring original evaluation function...")
            trainer.evaluate = original_eval_fn
        
        # 학습 종료 후 validation 실행 (항상 실행, 모델 저장 실패해도 실행)
        try:
            logger.info("\n" + "=" * 80)
            logger.info("🚀 Starting Post-Training Validation")
            logger.info("=" * 80)
            logger.info("⚠️ Note: Validation will run even if training was interrupted or model save failed")
            
            model_path = training_args.output_dir
            training_config_path = config_path
            
            # 모델이 저장되었는지 확인
            if not model_saved:
                logger.warning("⚠️ Model save failed, but validation will still attempt to run")
                logger.warning("⚠️ If validation fails, check if model files exist in output directory")
            
            # Config 파일 경로 찾기
            if training_config_path is None:
                # 기본 경로 시도
                default_config = "seqorth_sft/config/seqorth_small_config.json"
                if os.path.exists(default_config):
                    training_config_path = default_config
                    logger.debug(f"📄 Using default config: {default_config}")
                else:
                    logger.warning("⚠️ Training config path not found, some validations may be skipped")
            
            validation_results = run_post_training_validation(
                model_path=model_path,
                training_config_path=training_config_path,
                output_dir=training_args.output_dir,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            logger.info("✅ Post-training validation completed")
            
        except Exception as e:
            logger.error(f"❌ Post-training validation failed: {e}")
            log_error_context(logger, e, "post_training_validation")
            # Validation 실패해도 학습은 완료된 것으로 간주
            logger.error(f"❌ Validation error traceback:\n{traceback.format_exc()}")
        
        # WandB 명시적 종료 (로그 업로드 보장)
        try:
            # import wandb  # Global import used
            if wandb.run is not None:
                logger.info("👋 Finishing WandB run...")
                wandb.finish()
                logger.info("✅ WandB run finished")
        except Exception as wb_e:
            logger.warning(f"⚠️ Failed to finish WandB run: {wb_e}")


if __name__ == "__main__":
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Seqorth SFT Training with Config File")
        parser.add_argument(
            "--config", 
            type=str, 
            default="seqorth_sft/config/seqorth_small_config.json",
            help="Path to training configuration JSON file"
        )
        parser.add_argument(
            "--baseline",
            action="store_true",
            help="Seqorth 비활성화, Qwen3 원래 MoE만 사용 (backward 에러 회피)"
        )
        # Use parse_known_args to ignore --local_rank and other unknown args from DeepSpeed/Accelerate
        args, unknown = parser.parse_known_args()
        
        # --baseline 이면 환경 변수 설정 (자식 프로세스 env 미전달 문제 회피)
        if args.baseline:
            os.environ["SEQORTH_DISABLE_ALL"] = "1"
        
        # Load configuration
        config = load_config(args.config)
        
        model_config = config["model_config"]
        data_config = config["data_config"]
        training_config = config["training_config"]
        
        # Set seed
        set_seed(training_config["seed"])
        # wandb.init()은 Trainer가 자동으로 초기화하도록 함
        # DeepSpeed가 Trainer를 초기화할 때 wandb를 재초기화할 수 있으므로
        # 여기서 수동으로 초기화하지 않고 Trainer의 자동 초기화를 사용
        
        main(model_config, data_config, training_config, config_path=args.config)

    except Exception as e:
        logger.error(f"❌ Fatal error in main: {str(e)}")
        log_error_context(logger, e, "main_function")
        
        # OOM 에러인 경우 상세 정보 수집 및 저장
        error_msg = str(e)
        is_memory_error = (
            "CUDA out of memory" in error_msg or
            "CUBLAS_STATUS_ALLOC_FAILED" in error_msg or
            "cublasCreate" in error_msg
        )
        
        if is_memory_error:
            logger.error("❌ Fatal OOM error detected in main function")
            logger.error("💾 Collecting and saving error information...")
            try:
                # trainer 객체가 있는지 확인 (없을 수도 있음)
                trainer = None
                if 'trainer' in locals():
                    trainer = locals()['trainer']
                elif 'trainer' in globals():
                    trainer = globals()['trainer']
                
                if trainer is not None:
                    output_dir = None
                    if hasattr(trainer, 'args') and hasattr(trainer.args, 'output_dir'):
                        output_dir = trainer.args.output_dir
                    elif hasattr(trainer, 'training_args') and hasattr(trainer.training_args, 'output_dir'):
                        output_dir = trainer.training_args.output_dir
                    
                    error_file = save_oom_error_info(logger, trainer, e, batch_info=None, output_dir=output_dir)
                    if error_file:
                        logger.error(f"✅ Fatal OOM 에러 정보가 저장되었습니다: {error_file}")
                else:
                    # trainer가 없는 경우에도 환경 정보만이라도 저장
                    logger.error("⚠️ Trainer 객체를 찾을 수 없어 환경 정보만 수집합니다...")
                    try:
                        env_info = collect_environment_info()
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        error_file = f"logs/fatal_oom_error_info_{timestamp}.json"
                        os.makedirs("logs", exist_ok=True)
                        with open(error_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                'timestamp': timestamp,
                                'environment': env_info,
                                'error': {
                                    'error_type': type(e).__name__,
                                    'error_message': str(e),
                                    'error_traceback': traceback.format_exc()
                                }
                            }, f, indent=2, ensure_ascii=False, default=str)
                        logger.error(f"✅ Fatal OOM 에러 정보가 저장되었습니다: {error_file}")
                    except Exception as save_e:
                        logger.error(f"❌ 에러 정보 저장 실패: {save_e}")
            except Exception as collect_e:
                logger.error(f"❌ 에러 정보 수집 실패: {collect_e}")
        
        # Log final memory state
        if torch.cuda.is_available():
            logger.error("❌ Final GPU memory state:")
            logger.error(f"❌ Memory summary:\n{torch.cuda.memory_summary()}")
            logger.error(f"❌ Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            logger.error(f"❌ Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f}GB")
        
        # Re-raise the exception
        raise e
