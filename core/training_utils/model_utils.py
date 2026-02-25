"""
Model utilities for training
"""
import logging
from typing import Optional, Dict, Any, Set
import torch


def get_vision_parameter_ids(model) -> Set[int]:
    """
    Get set of parameter IDs for all vision-related parameters.
    Used to protect vision freeze from being undone by other utilities.
    
    Returns:
        Set of parameter IDs (memory addresses) for vision parameters
    """
    vision_param_ids = set()
    
    # Method 1: Use vision_tower property if available
    vision_tower = None
    try:
        actual_model = model.module if hasattr(model, 'module') else model
        if hasattr(actual_model, 'vision_tower'):
            vision_tower = actual_model.vision_tower
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'vision_tower'):
            vision_tower = actual_model.model.vision_tower
        elif hasattr(actual_model, 'visual'):
            vision_tower = actual_model.visual
    except:
        pass
    
    if vision_tower is not None:
        for param in vision_tower.parameters():
            vision_param_ids.add(id(param))
    else:
        # Method 2: Find vision modules by name and class
        vision_module_names = ['visual', 'vision_tower', 'vision_model', 'vision_encoder']
        for name, module in model.named_modules():
            # Check if this is a vision module by name
            is_vision_module = False
            for vision_name in vision_module_names:
                if vision_name in name.lower() or name.startswith(vision_name):
                    is_vision_module = True
                    break
            
            # Also check by class name
            if not is_vision_module:
                class_name = type(module).__name__
                if any(v in class_name for v in ['Vision', 'Siglip', 'CLIP', 'Qwen3VLVision', 'Qwen2VLVision']):
                    is_vision_module = True
            
            if is_vision_module:
                for param in module.parameters():
                    vision_param_ids.add(id(param))
    
    return vision_param_ids


def protect_vision_freeze(model, logger: Optional[logging.Logger] = None) -> int:
    """
    Protect vision freeze by ensuring all vision parameters remain frozen.
    Call this after any operation that might modify requires_grad.
    
    Returns:
        Number of vision parameters that were re-frozen
    """
    vision_param_ids = get_vision_parameter_ids(model)
    
    if not vision_param_ids:
        if logger:
            logger.debug("  ℹ️  No vision parameters found to protect")
        return 0
    
    refrozen_count = 0
    actual_model = model.module if hasattr(model, 'module') else model
    
    for name, param in actual_model.named_parameters():
        if id(param) in vision_param_ids:
            if param.requires_grad:
                param.requires_grad = False
                refrozen_count += 1
                if logger:
                    logger.debug(f"  🔒 Re-froze vision parameter: {name}")
    
    if logger and refrozen_count > 0:
        logger.info(f"  ✅ Protected vision freeze: re-froze {refrozen_count} vision parameters")
    
    return refrozen_count


def count_unique_parameters(model, verbose: bool = False, logger: Optional[logging.Logger] = None) -> int:
    """
    Count parameters accurately, avoiding shared module duplication.
    Shared modules (e.g., global router across layers) are counted only once.
    """
    seen_params = set()
    total_params = 0
    param_names_by_ptr = {}  # For debugging: track which params share memory
    
    for name, param in model.named_parameters():
        # Use param.data_ptr() to identify unique tensors (shared params have same ptr)
        param_id = param.data_ptr()
        if param_id not in seen_params:
            seen_params.add(param_id)
            total_params += param.numel()
            param_names_by_ptr[param_id] = [name]
        else:
            # This parameter shares memory with another (e.g., shared router)
            param_names_by_ptr[param_id].append(name)
    
    if verbose and logger:
        # Report shared parameters
        shared_groups = {ptr: names for ptr, names in param_names_by_ptr.items() if len(names) > 1}
        if shared_groups:
            logger.info(f"  📊 Found {len(shared_groups)} groups of shared parameters:")
            for ptr, names in list(shared_groups.items())[:3]:  # Show first 3 groups
                logger.info(f"     - {len(names)} params sharing memory: {names[0]} (+ {len(names)-1} others)")
            if len(shared_groups) > 3:
                logger.info(f"     ... and {len(shared_groups) - 3} more shared groups")
    
    return total_params


def check_model_size(model, logger: logging.Logger, context: str = "model_check") -> Optional[Dict[str, Any]]:
    """
    상세한 모델 크기 검사 함수
    모델의 전체 크기, 컴포넌트별 크기, 메모리 사용량 등을 검사
    """
    from core.training_utils.utils import format_parameters
    from core.training_utils.logging_utils import log_gpu_memory
    import traceback
    
    logger.info("=" * 80)
    logger.info(f"🔍 Model Size Check: {context}")
    logger.info("=" * 80)
    
    try:
        # 전체 파라미터 카운팅
        total_params = count_unique_parameters(model, verbose=False, logger=logger)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        naive_total = sum(p.numel() for p in model.parameters())

        # Debug: Check if we have the expected number of parameters for Qwen3-VL-30B
        expected_params = 30_000_000_000  # 30B parameters
        if total_params < expected_params * 0.1:  # Less than 10% of expected
            logger.warning(f"⚠️  Parameter count seems too low: {format_parameters(total_params)}")
            logger.warning(f"    Expected ~{format_parameters(expected_params)} for Qwen3-VL-30B")
            logger.warning(f"    This might indicate model loading issues or excessive LoRA pruning")

            # Debug: Count parameters by component type
            debug_total = 0
            for name, param in model.named_parameters():
                if 'qwen3' in name.lower() or 'vl' in name.lower():
                    debug_total += param.numel()
            logger.warning(f"    Qwen3/VL related params found: {format_parameters(debug_total)}")
        
        logger.info(f"📊 Total Parameters:")
        logger.info(f"  - Unique (deduplicated): {format_parameters(total_params)}")
        if naive_total != total_params:
            logger.info(f"  - Naive count (with duplicates): {format_parameters(naive_total)}")
            logger.info(f"  - Deduplication saved: {format_parameters(naive_total - total_params)} ({(naive_total - total_params)/naive_total*100:.1f}%)")
        if total_params > 0:
            logger.info(f"  - Trainable: {format_parameters(trainable_params)} ({trainable_params/total_params*100:.2f}%)")
        else:
            logger.info(f"  - Trainable: {format_parameters(trainable_params)} (N/A %)")
        if total_params > 0:
            logger.info(f"  - Frozen: {format_parameters(total_params - trainable_params)} ({(total_params - trainable_params)/total_params*100:.2f}%)")
        else:
            logger.info(f"  - Frozen: {format_parameters(total_params - trainable_params)} (N/A %)")
        
        # 컴포넌트별 파라미터 카운팅
        component_params = {}
        try:
            # Vision tower
            if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
                vision_params = sum(p.numel() for p in model.model.vision_tower.parameters())
                component_params['vision_tower'] = vision_params
            elif hasattr(model, 'vision_tower'):
                vision_params = sum(p.numel() for p in model.vision_tower.parameters())
                component_params['vision_tower'] = vision_params
            
            # Language model
            if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
                lang_params = sum(p.numel() for p in model.model.language_model.parameters())
                component_params['language_model'] = lang_params
            elif hasattr(model, 'language_model'):
                lang_params = sum(p.numel() for p in model.language_model.parameters())
                component_params['language_model'] = lang_params
            
            # Router parameters (use unique counting to avoid duplicates from shared routers)
            router_params = 0
            router_count = 0
            router_param_ptrs = set()  # Track unique parameter pointers
            from models.spectra_model import SPECTRARouter
            for name, module in model.named_modules():
                if isinstance(module, SPECTRARouter):
                    router_count += 1
                    # Count unique parameters only (shared routers have same param pointers)
                    for p in module.parameters():
                        param_ptr = p.data_ptr()
                        if param_ptr not in router_param_ptrs:
                            router_param_ptrs.add(param_ptr)
                            router_params += p.numel()
            if router_count > 0:
                component_params['routers'] = router_params
                logger.info(f"  - Router modules found: {router_count} (unique params: {format_parameters(router_params)})")
                
                # Detailed router breakdown
                if router_count == 1:
                    # Single router - show detailed breakdown
                    router = next((m for m in model.modules() if isinstance(m, SPECTRARouter)), None)
                    if router:
                        router_details = {}
                        if hasattr(router, 'load_balancer'):
                            balancer_params = sum(p.numel() for p in router.load_balancer.parameters())
                            router_details['load_balancer'] = balancer_params
                        if hasattr(router, 'bias_proj'):
                            bias_proj_params = sum(p.numel() for p in router.bias_proj.parameters())
                            router_details['bias_proj'] = bias_proj_params
                        if hasattr(router, 'expression_projector'):
                            expr_proj_params = sum(p.numel() for p in router.expression_projector.parameters())
                            router_details['expression_projector'] = expr_proj_params
                        
                        if router_details:
                            logger.info(f"  📋 Router component breakdown:")
                            for comp_name, comp_params in router_details.items():
                                logger.info(f"     - {comp_name}: {format_parameters(comp_params)}")
                else:
                    logger.warning(f"  ⚠️  Multiple router instances found ({router_count}) - may indicate duplication issue")
            
            # MoE experts (including Qwen3 VL MoE experts)
            expert_params = 0
            expert_count = 0
            qwen_expert_params = 0
            qwen_expert_count = 0

            from models.spectra_model import SPECTRAMoE
            for name, module in model.named_modules():
                if isinstance(module, SPECTRAMoE):
                    expert_count += 1
                    expert_params += sum(p.numel() for p in module.parameters())

                # Also count Qwen3 VL MoE experts directly
                if hasattr(module, '__class__') and 'Qwen3VLMoeTextExperts' in module.__class__.__name__:
                    qwen_expert_count += 1
                    qwen_expert_params += sum(p.numel() for p in module.parameters())

            if expert_count > 0:
                component_params['moe_experts'] = expert_params
                logger.info(f"  - SPECTRA MoE modules found: {expert_count}")

            if qwen_expert_count > 0:
                component_params['qwen_experts'] = qwen_expert_params
                logger.info(f"  - Qwen3 VL MoE experts found: {qwen_expert_count} (params: {format_parameters(qwen_expert_params)})")
            
            if component_params:
                logger.info(f"📦 Component Breakdown:")
                for comp_name, comp_params in component_params.items():
                    logger.info(f"  - {comp_name}: {format_parameters(comp_params)} ({comp_params/total_params*100:.2f}%)")
        except Exception as e:
            logger.debug(f"  Component breakdown failed: {e}")
        
        # 메모리 사용량
        if torch.cuda.is_available():
            memory_info = log_gpu_memory(logger, context)
            if memory_info:
                logger.info(f"💾 GPU Memory:")
                logger.info(f"  - Allocated: {memory_info.get('allocated', 0):.2f}GB")
                logger.info(f"  - Reserved: {memory_info.get('reserved', 0):.2f}GB")
        
        # 모델 dtype 및 device 정보
        try:
            first_param = next(model.parameters())
            logger.info(f"⚙️  Model Configuration:")
            logger.info(f"  - Dtype: {first_param.dtype}")
            logger.info(f"  - Device: {first_param.device}")
        except:
            pass
        
        logger.info("=" * 80)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'naive_total': naive_total,
            'component_params': component_params
        }
    
    except Exception as e:
        logger.error(f"❌ Model size check failed: {e}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return None


def get_dynamic_lora_target_modules(model, logger: logging.Logger):
    """
    Dynamically identify LoRA target modules by scanning the model structure.
    Only returns modules that actually exist in the model.
    Focuses on experts, attention, and router components.
    Excludes vision-related modules.
    
    CRITICAL: For large models with many experts (e.g., 128 experts in Qwen3-VL-30B),
    scanning all modules causes RAM explosion. We only scan the FIRST MoE instance
    to detect the expert structure, then validate against actual model structure.
    """
    from models.spectra_model import SPECTRAMoE, SPECTRARouter
    import torch.nn as nn
    import re
    
    # First, collect all actual module names and modules from the model
    actual_model = model.module if hasattr(model, 'module') else model
    all_module_names = [name for name, _ in actual_model.named_modules()]
    
    # Vision-related prefixes to exclude
    vision_prefixes = ['visual', 'vision_tower', 'vision_model', 'vision_encoder', 'patch_embed', 'pos_embed']
    
    target_modules = set()
    target_module_names = set()  # Track full module names for validation
    
    # 1. Direct scan: Find all Linear modules that are NOT vision-related
    logger.debug("  🔍 Scanning model for Linear modules...")
    for module_name, module in actual_model.named_modules():
        # Skip vision-related modules
        if any(prefix in module_name.lower() for prefix in vision_prefixes):
            continue
        
        # Check if this is a Linear module
        if isinstance(module, nn.Linear):
            leaf_name = module_name.split('.')[-1]
            if leaf_name and not leaf_name.isdigit():
                # Exclude common non-target modules
                exclude_keywords = ['norm', 'ln_', 'layernorm', 'embed', 'lm_head', 'classification_head']
                if not any(exclude in leaf_name.lower() for exclude in exclude_keywords):
                    target_modules.add(leaf_name)
                    target_module_names.add(module_name)
                    logger.debug(f"  ✅ Found Linear module: {module_name} -> {leaf_name}")
    
    # 2. Pattern-based matching for known projection patterns (only if not found above)
    if not target_modules:
        logger.debug("  🔍 No Linear modules found, trying pattern matching...")
        base_target_patterns = [
            # Qwen3-VL MLP modules (inside mlp) - regex patterns
            r"layers\.\d+\.mlp\.gate_proj",
            r"layers\.\d+\.mlp\.up_proj",
            r"layers\.\d+\.mlp\.down_proj",
            r"layers\.\d+\.mlp\.gete_up_proj",
            # Standard FFN projections (suffix match)
            r"\.gate_proj$",
            r"\.up_proj$",
            r"\.down_proj$",
            r"\.gate_up_proj$",
            r"\.gate_up_proj_bias$",
            r"\.down_proj_bias$",
            # GRU modules
            r"\.weight_ih_gates$",
            r"\.weight_hh_gates$",
            r"\.weight_ih_cand$",
            r"\.weight_hh_cand$",
            # Router/Heads (suffix match)
            r"\.bias_proj$",
            r"\.linear_projection$",
            r"\.projection_head$",
            r"\.projection$",  # SPECTRA Router components
        ]
        
        # Match patterns against actual module names (excluding vision)
        for pattern in base_target_patterns:
            compiled_pattern = re.compile(pattern)
            for module_name in all_module_names:
                # Skip vision-related modules
                if any(prefix in module_name.lower() for prefix in vision_prefixes):
                    continue
                
                if compiled_pattern.search(module_name):
                    # Extract the leaf module name (last component)
                    leaf_name = module_name.split('.')[-1]
                    if leaf_name and not leaf_name.isdigit():
                        target_modules.add(leaf_name)
                        target_module_names.add(module_name)
                        logger.debug(f"  ✅ Found matching module: {module_name} -> {leaf_name}")
    
    # 3. Detect expert structure from FIRST MoE instance only (to prevent RAM explosion)
    monolithic_expert_found = False
    parameter_based_expert_found = False
    first_moe_scanned = False
    
    for name, module in actual_model.named_modules():
        # Skip vision-related modules
        if any(prefix in name.lower() for prefix in vision_prefixes):
            continue
        
        # Only scan the FIRST SPECTRAMoE instance to detect structure
        if isinstance(module, SPECTRAMoE) and not first_moe_scanned:
            first_moe_scanned = True
            
            if hasattr(module, 'experts') and len(module.experts) > 0:
                first_expert = module.experts[0]
                expert_class_name = first_expert.__class__.__name__
                
                # Detect monolithic expert classes
                is_monolithic = (
                    'Experts' in expert_class_name or
                    'MoE' in expert_class_name or
                    (hasattr(first_expert, 'num_experts') and first_expert.num_experts > 1) or
                    (hasattr(first_expert, 'n_experts') and first_expert.n_experts > 1)
                )
                
                if is_monolithic:
                    monolithic_expert_found = True
                    logger.debug(f"  🔍 Detected monolithic expert class: {expert_class_name}")
                    
                    # Check if parameter-based (nn.Parameter) or module-based (nn.Linear)
                    expert_params = dict(first_expert.named_parameters(recurse=False))
                    if expert_params:
                        parameter_based_expert_found = True
                        logger.debug(f"     📦 Parameter-based expert (uses nn.Parameter)")
                        
                        # Add parameter names to targets (but validate they exist in model)
                        for param_name in expert_params.keys():
                            if any(keyword in param_name.lower() for keyword in ['proj', 'weight', 'bias']):
                                if not any(exclude in param_name.lower() for exclude in ['norm', 'ln_', 'layernorm']):
                                    # Check if this parameter name exists in any module
                                    found_in_model = any(
                                        (param_name in mod_name or mod_name.endswith(f".{param_name}"))
                                        and not any(prefix in mod_name.lower() for prefix in vision_prefixes)
                                        for mod_name in all_module_names
                                    )
                                    if found_in_model:
                                        target_modules.add(param_name)
                                        logger.debug(f"       Found parameter: {param_name}")
                    else:
                        # Module-based expert - scan for Linear modules (limit depth)
                        logger.debug(f"     🔧 Module-based expert (uses nn.Linear)")
                        
                        for sub_name, sub_module in first_expert.named_modules():
                            # Limit depth to prevent excessive scanning
                            if sub_name.count('.') > 2:
                                continue
                                
                            cls_name = sub_module.__class__.__name__
                            if "Linear" in cls_name:
                                leaf_name = sub_name.split('.')[-1]
                                if leaf_name and not leaf_name.isdigit():
                                    if any(keyword in leaf_name.lower() for keyword in ['proj', 'weight', 'bias', 'linear']):
                                        # Validate this module name exists in the actual model
                                        found_in_model = any(
                                            (leaf_name in mod_name or mod_name.endswith(f".{leaf_name}"))
                                            and not any(prefix in mod_name.lower() for prefix in vision_prefixes)
                                            for mod_name in all_module_names
                                        )
                                        if found_in_model:
                                            target_modules.add(leaf_name)
                                            logger.debug(f"       Found module: {leaf_name}")
            
            # Stop after first MoE to prevent RAM explosion
            break
    
    # 4. Final validation: Only keep targets that actually exist in the model (excluding vision)
    validated_targets = set()
    exclude_list = {"norm", "ln_f", "ln_1", "ln_2", "embed", "lm_head", "classification_head", "layernorm", "gate_proj", "up_proj", "down_proj"}
    
    for target in target_modules:
        if target.lower() in exclude_list:
            continue
        
        # Check if this target exists in the model (suffix match or contains match), excluding vision
        found = False
        for module_name in all_module_names:
            # Skip vision-related modules
            if any(prefix in module_name.lower() for prefix in vision_prefixes):
                continue
            
            # Suffix match (e.g., "gate_proj" matches "model.layers.0.mlp.gate_proj")
            if module_name.endswith(f".{target}") or module_name == target:
                found = True
                break
            # Also check if target is a substring (for patterns like "projection")
            if f".{target}." in f".{module_name}." or module_name.endswith(f".{target}"):
                found = True
                break
        
        if found:
            validated_targets.add(target)
        else:
            logger.debug(f"  ⚠️  Target '{target}' not found in model (or is vision-related), skipping")
    
    final_targets = sorted(validated_targets)
    
    if monolithic_expert_found:
        if parameter_based_expert_found:
            logger.info(f"✅ Parameter-based monolithic experts detected (nn.Parameter projections)")
            logger.warning(f"⚠️  PEFT cannot apply LoRA to nn.Parameter - experts will remain frozen")
        else:
            logger.info(f"✅ Module-based monolithic experts detected (nn.Linear projections)")
    
    if not final_targets:
        logger.warning(f"⚠️  No LoRA target modules found in model structure!")
        logger.debug(f"   First 20 module names: {all_module_names[:20]}")
        # Try to find language model part
        language_model_paths = [name for name in all_module_names if 'language_model' in name.lower() or ('layers' in name.lower() and 'visual' not in name.lower())]
        if language_model_paths:
            logger.debug(f"   Found {len(language_model_paths)} potential language model modules (first 10): {language_model_paths[:10]}")
    else:
        logger.info(f"🚀 Dynamically identified {len(final_targets)} LoRA target modules:")
        logger.info(f"   {final_targets}")
    
    return final_targets


def verify_model_consistency(model, logger: logging.Logger, output_dir: str = "."):
    """
    Verify that all ranks have the exact same trainable parameters.
    This prevents NCCL hangs/crashes at Step 0 due to attribute mismatches.
    """
    import torch.distributed as dist
    import os
    
    if not (dist.is_available() and dist.is_initialized()):
        logger.info("  ℹ️  Distributed not initialized, skipping consistency check")
        return

    logger.info("🔍 Verifying model parameter consistency across ranks...")
    
    # 1. Compute local checksum of (name, requires_grad)
    # We use a simple hash of the string representation
    local_params_state = []
    actual_model = model.module if hasattr(model, 'module') else model
    
    for name, param in actual_model.named_parameters():
        state = f"{name}:{param.requires_grad}:{param.shape}"
        local_params_state.append(state)
    
    # Sort to ensure order
    local_params_state.sort()
    full_state_str = "|".join(local_params_state)
    
    # Simple hash (adler32 is fast and sufficient for this)
    import zlib
    local_hash = zlib.adler32(full_state_str.encode('utf-8'))
    
    # CRITICAL: NCCL backend does not support CPU tensors.
    # We must use CUDA device for all_gather when NCCL is the backend.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    local_tensor = torch.tensor([local_hash], device=device, dtype=torch.long)
    
    # 2. Compare with other ranks
    # We gather all hashes to rank 0 (or all_gather to everyone)
    world_size = dist.get_world_size()
    all_hashes = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(all_hashes, local_tensor)
    
    # 3. Check for mismatches
    rank = dist.get_rank()
    mismatches_found = False
    ref_hash = all_hashes[0].item()
    
    for r, h in enumerate(all_hashes):
        if h.item() != ref_hash:
            mismatches_found = True
            logger.error(f"❌ Consistency Check Failed: Rank {r} hash ({h.item()}) != Rank 0 hash ({ref_hash})")
    
    if mismatches_found:
        # Dump state for debugging
        params_dump_file = os.path.join(output_dir, f"consistency_dump_rank_{rank}.txt")
        with open(params_dump_file, "w") as f:
            for s in local_params_state:
                f.write(s + "\n")
        logger.error(f"❌ Mismatch detected! Parameter state dumped to {params_dump_file}")
        
        # Force barrier to ensure all ranks log before crashing
        try:
            dist.barrier()
        except:
            pass
            
        raise RuntimeError("CRITICAL: Model parameter mismatch detected across ranks! Training aborted to prevent NCCL hang.")
    else:
        logger.info("✅ Model parameter consistency check passed (all ranks match).")



