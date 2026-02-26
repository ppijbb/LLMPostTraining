#!/usr/bin/env python3
"""
Fast DeepSpeed Forward/Backward Test with Dummy Data
Tests Seqorth model with DeepSpeed ZeRO-3 without loading actual dataset.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
import deepspeed
from pathlib import Path
import logging
import atexit

# Suppress TensorFlow/grpc warnings and errors
# These occur during process shutdown and don't affect training
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress INFO, WARNING, ERROR
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")  # Only show errors, not warnings
os.environ.setdefault("GLOG_minloglevel", "2")  # Suppress glog INFO/WARNING

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_deepspeed_zero3_linear_bias_dtype() -> None:
    """
    Patch DeepSpeed ZeRO-3 linear wrapper to prevent BF16/Half mismatch inside torch.addmm.

    Observed failure:
      RuntimeError: self and mat2 must have the same dtype, but got BFloat16 and Half

    This can occur when bias is BF16 but gathered ZeRO-3 weight shard is FP16 (or vice-versa).
    We align bias dtype to weight dtype right before the matmul.
    """
    try:
        import deepspeed.runtime.zero.linear as ds_zero_linear
        from deepspeed.runtime.zero.linear import autocast_custom_fwd
        
        if getattr(ds_zero_linear, "_seqorth_bias_dtype_patch_applied", False):
            return

        LinearFn = getattr(ds_zero_linear, "LinearFunctionForZeroStage3", None)
        if LinearFn is None or not hasattr(LinearFn, "forward"):
            logger.warning("⚠️ Could not find LinearFunctionForZeroStage3.forward to patch")
            return

        # We MUST replace the entire method with a NEW decorated one to ensure 
        # our dtype alignment happens AFTER any autocast logic.
        
        original_forward = LinearFn.forward
        if hasattr(original_forward, "__wrapped__"):
            # If it was already decorated, we might want the original original
            pass

        @staticmethod
        @autocast_custom_fwd
        def patched_forward(ctx, input, weight, *args, **kwargs):
            """
            DeepSpeed versions differ in whether `bias` is passed to forward() when None.
            """
            bias = args[0] if len(args) >= 1 else None

            if weight is not None:
                try:
                    # Gathered weight dtype is the source of truth for the matmul
                    w_dtype = getattr(weight, "dtype", None)
                    if w_dtype is not None:
                        # Force input and bias to match weight dtype exactly
                        if getattr(input, "dtype", None) != w_dtype:
                            input = input.to(dtype=w_dtype)
                        if bias is not None and getattr(bias, "dtype", None) != w_dtype:
                            bias = bias.to(dtype=w_dtype)
                except Exception as e:
                    pass

            # Call the ORIGINAL UNDECORATED forward if we can find it, 
            # but since we don't have it easily, we call a version that doesn't re-decorate.
            # Actually, the simplest is to just perform the logic here if we're already decorated.
            
            # Re-implementing the simple forward logic from DeepSpeed to be safe
            ctx.save_for_backward(input, weight, bias)

            if input.dim() == 2 and bias is not None:
                return torch.addmm(bias, input, weight.t())
            else:
                output = input.matmul(weight.t())
                if bias is not None:
                    output += bias
                return output

        LinearFn.forward = patched_forward
        ds_zero_linear._seqorth_bias_dtype_patch_applied = True
        logger.info("✅ Patched DeepSpeed ZeRO-3 LinearFunctionForZeroStage3.forward (with @autocast_custom_fwd) to align dtypes")
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply DeepSpeed ZeRO-3 linear bias dtype patch: {e}")


def cleanup_grpc():
    """Cleanup grpc resources on exit to prevent timeout warnings."""
    try:
        # Try to properly shutdown any grpc resources
        import grpc
        # Force shutdown any remaining channels
        # This is a best-effort cleanup
        pass
    except ImportError:
        pass
    except Exception:
        # Ignore any errors during cleanup
        pass

# Register cleanup function
atexit.register(cleanup_grpc)

# Apply DeepSpeed ZeRO-3 linear dtype patch at import time (all ranks)
patch_deepspeed_zero3_linear_bias_dtype()

# Add parent directory to path to import train_seqorth
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Seqorth model
from models.seqorth_model import SeqorthForConditionalGeneration, SeqorthConfig
from transformers import AutoConfig, AutoTokenizer

# Import setup_model from train_seqorth
from training.seqorth_sft.train_seqorth import setup_model


def setup_distributed():
    """Initialize distributed training with extreme timeouts for ZeRO-3 NVMe offload."""
    if not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            
            torch.cuda.set_device(local_rank)
            
            # CRITICAL: Extreme timeout for 30B+ MoE model with NVMe offload
            # Sharding and offloading 60GB+ of parameters can take >30 mins
            from datetime import timedelta
            timeout = timedelta(minutes=120) 
            
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timeout
            )
            logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
            logger.info(f"NCCL Timeout set to {timeout}")
            
            # Set DeepSpeed environment variables for timeout
            os.environ["DEEPSPEED_TIMEOUT"] = "7200" # 2 hours
            
            from transformers.trainer_utils import set_seed
            set_seed(4008)
            logger.info("Fixed random seed to 4008 for all ranks.")
        else:
            # DeepSpeed ZeRO Init requires torch.distributed to be initialized.
            # If we don't initialize, DeepSpeed will try MPI discovery (mpi4py/libmpi)
            # which is often unavailable in containerized environments.
            #
            # So for a single-GPU run, we initialize a 1-rank process group via TCP/NCCL.
            if torch.cuda.is_available():
                import socket
                from datetime import timedelta

                def _find_free_port() -> int:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("127.0.0.1", 0))
                        return int(s.getsockname()[1])

                port = os.environ.get("MASTER_PORT")
                if port is None:
                    port = str(_find_free_port())
                    os.environ["MASTER_PORT"] = port
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")
                os.environ.setdefault("LOCAL_RANK", "0")

                # torch.cuda.set_device(0)
                timeout = timedelta(minutes=120)
                init_kwargs = dict(
                    backend="nccl",
                    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                    rank=0,
                    world_size=1,
                    timeout=timeout,
                )
                # Torch supports explicit device_id (avoids NCCL hang/warnings when rank->GPU mapping is implicit)
                try:
                    dist.init_process_group(**init_kwargs, device_id=torch.device("cuda:0"))
                except TypeError:
                    dist.init_process_group(**init_kwargs)
                logger.info(
                    "Initialized single-process distributed group for DeepSpeed: "
                    f"rank=0 world_size=1 master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
                )
                return True

            logger.info("Running in single GPU mode (no distributed, CUDA not available)")
            return False
    return True


def verify_rank_consistency(data, logger):
    """Check if tensors are identical across all ranks."""
    if not dist.is_initialized():
        return True
    
    # Get device from CUDA
    device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
    
    is_consistent = True
    for key, val in data.items():
        if torch.is_tensor(val):
            # Move to GPU and use a simple sum as a proxy for identity
            local_val = val.float().to(device).sum()
            all_vals = [torch.zeros_like(local_val).to(device) for _ in range(dist.get_world_size())]
            dist.all_gather(all_vals, local_val)
            
            for i, remote_val in enumerate(all_vals):
                if not torch.allclose(local_val, remote_val, atol=1e-5):
                    logger.error(f"❌ [DESYNC DETECTED] Rank {dist.get_rank()} vs Rank {i} for key '{key}': local={local_val.item():.6f}, remote={remote_val.item():.6f}")
                    is_consistent = False
                    break
    
    if is_consistent:
        logger.info("✅ All ranks have identical input data.")
    return is_consistent


def load_configs(config_path: str, deepspeed_config_path: str, batch_size: int = 1):
    """Load model and DeepSpeed configurations."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    with open(deepspeed_config_path, 'r') as f:
        ds_config_dict = json.load(f)
    
    # CRITICAL FIX: Replace "auto" values with actual numbers for deepspeed.initialize()
    # DeepSpeed's initialize() doesn't handle "auto" like Trainer does
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = ds_config_dict.get("gradient_accumulation_steps", 1)
    if isinstance(gradient_accumulation_steps, str) and gradient_accumulation_steps == "auto":
        gradient_accumulation_steps = 1
    
    # CRITICAL: Read SP size from Accelerate config for batch size calculation
    sp_size = ds_config_dict.get("sequence_parallel_size", 1)
    if sp_size == 1:
        try:
            import yaml
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent.parent
            accelerate_config_path = project_root / "training" / "seqorth_sft" / "config" / "accelerate.yaml"
            if accelerate_config_path.exists():
                with open(accelerate_config_path, 'r') as f:
                    accelerate_config = yaml.safe_load(f)
                    parallelism_config = accelerate_config.get("parallelism_config", {})
                    sp_size = parallelism_config.get("sp_size", 1)
                    if sp_size > 1:
                        ds_config_dict["sequence_parallel_size"] = sp_size
                        logger.info(f"✅ Applied SP size {sp_size} from Accelerate config to DeepSpeed config")
        except Exception as e:
            logger.warning(f"⚠️ Failed to read SP size from Accelerate config: {e}, using SP size 1")
    
    if "train_batch_size" in ds_config_dict and ds_config_dict["train_batch_size"] == "auto":
        # CRITICAL: When SP is enabled, DeepSpeed expects:
        # train_batch_size = micro_batch_size * gradient_accumulation_steps * (world_size / sp_size)
        # Without SP: train_batch_size = micro_batch_size * gradient_accumulation_steps * world_size
        if sp_size > 1:
            effective_world_size = world_size // sp_size
            ds_config_dict["train_batch_size"] = batch_size * gradient_accumulation_steps * effective_world_size
            logger.info(f"✅ [SP] Calculated train_batch_size with SP: {batch_size} * {gradient_accumulation_steps} * {effective_world_size} = {ds_config_dict['train_batch_size']} (sp_size={sp_size})")
        else:
            ds_config_dict["train_batch_size"] = batch_size * gradient_accumulation_steps * world_size
            logger.info(f"✅ Calculated train_batch_size: {batch_size} * {gradient_accumulation_steps} * {world_size} = {ds_config_dict['train_batch_size']}")
    
    if "train_micro_batch_size_per_gpu" in ds_config_dict and ds_config_dict["train_micro_batch_size_per_gpu"] == "auto":
        ds_config_dict["train_micro_batch_size_per_gpu"] = batch_size
    
    if "gradient_accumulation_steps" in ds_config_dict and ds_config_dict["gradient_accumulation_steps"] == "auto":
        ds_config_dict["gradient_accumulation_steps"] = gradient_accumulation_steps
    
    if "gradient_clipping" in ds_config_dict and ds_config_dict["gradient_clipping"] == "auto":
        ds_config_dict["gradient_clipping"] = 1.0
    
    # Handle scheduler "auto" values
    if "scheduler" in ds_config_dict and isinstance(ds_config_dict["scheduler"], dict):
        scheduler_params = ds_config_dict["scheduler"].get("params", {})
        for key in ["total_num_steps", "warmup_min_lr", "warmup_max_lr", "warmup_num_steps"]:
            if key in scheduler_params and scheduler_params[key] == "auto":
                if key == "total_num_steps":
                    scheduler_params[key] = 1000
                elif key == "warmup_num_steps":
                    scheduler_params[key] = 100
                elif key in ["warmup_min_lr", "warmup_max_lr"]:
                    scheduler_params[key] = 0.0
    
    return config_dict, ds_config_dict


def create_dummy_data(config, model=None, batch_size=1, seq_len=128, num_images=1):
    """Create dummy input data for testing using actual processor."""
    # Try to get vocab_size from model config if available, otherwise from config
    vocab_size = None
    if model is not None:
        try:
            # Get actual model (unwrap DeepSpeed/PEFT if needed)
            actual_model = model
            if hasattr(model, 'module'):
                actual_model = model.module
            if hasattr(actual_model, 'base_model'):
                actual_model = actual_model.base_model
            if hasattr(actual_model, 'model'):
                actual_model = actual_model.model
            
            # Try to get vocab_size from config (safer than accessing embedding layer with ZeRO-3)
            if hasattr(actual_model, 'config'):
                vocab_size = getattr(actual_model.config, 'vocab_size', None)
                # Also check text_config if it exists
                if vocab_size is None and hasattr(actual_model.config, 'text_config'):
                    vocab_size = getattr(actual_model.config.text_config, 'vocab_size', None)
        except Exception as e:
            logger.warning(f"Could not get vocab_size from model config: {e}")
    
    # Fallback to config
    if vocab_size is None:
        vocab_size = config["model_config"]["seqorth_params"].get("vocab_size", 151936)
        logger.info(f"Using vocab_size from config: {vocab_size}")
    else:
        logger.info(f"Using vocab_size from model config: {vocab_size}")
    
    image_token_id = config["model_config"].get("image_token_index", config["model_config"].get("image_token_id", 151655))
    
    # input_ids and attention_mask will be created by collate function if available
    # Otherwise create manually
    input_ids = None
    attention_mask = None
    
    # CRITICAL: Use actual processor and collate function to create pixel_values and image_grid_thw
    # This ensures the format matches exactly what the model expects (same as training)
    try:
        from transformers import AutoProcessor
        from PIL import Image
        from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
        
        processor_path = config["model_config"].get("tokenizer_name_or_path") or config["model_config"].get("model_name_or_path")
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=config["model_config"].get("trust_remote_code", True)
        )
        
        # Create dummy messages format (same as training data)
        dummy_messages_list = []
        for b in range(batch_size):
            # Create messages with images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"} for _ in range(num_images)
                    ] + [{"type": "text", "text": "dummy text"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "dummy response"}]
                }
            ]
            dummy_messages_list.append({"messages": messages, "images": [Image.new('RGB', (224, 224), color='black') for _ in range(num_images)]})
        
        # Use the same collate function as training
        collate_fn = DataCollatorForVisionLanguageModeling(
            processor=processor,
            max_length=seq_len
        )
        
        # Process using collate function (same as training)
        batch = collate_fn(dummy_messages_list)
        
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        
        if pixel_values is None:
            raise ValueError("Collate function did not return pixel_values")
        
        # If image_grid_thw is not in batch, create it from pixel_values shape
        if image_grid_thw is None:
            logger.warning("⚠️ Collate function did not return image_grid_thw, inferring from pixel_values...")
            # Infer from pixel_values shape
            if len(pixel_values.shape) == 2:
                # Flattened format [total_tokens, hidden_dim]
                total_tokens = pixel_values.shape[0]
                # Estimate grid size (assuming square patches)
                # For Qwen3-VL, typically uses 14x14 = 196 tokens per 224x224 image
                tokens_per_image = total_tokens // (batch_size * num_images) if batch_size * num_images > 0 else 196
                grid_size = int(tokens_per_image ** 0.5)  # Assume square grid
                image_grid_thw = torch.zeros(batch_size, 3, dtype=torch.long)
                for b in range(batch_size):
                    image_grid_thw[b] = torch.tensor([tokens_per_image * num_images, grid_size, grid_size])
            else:
                # Standard format - use default
                height_grid = 14
                width_grid = 14
                num_tokens_per_image = height_grid * width_grid
                image_grid_thw = torch.zeros(batch_size, 3, dtype=torch.long)
                for b in range(batch_size):
                    total_tokens = num_tokens_per_image * num_images
                    image_grid_thw[b] = torch.tensor([total_tokens, height_grid, width_grid])
        
        logger.info(f"✅ Created pixel_values with shape: {pixel_values.shape}")
        logger.info(f"✅ Created image_grid_thw with shape: {image_grid_thw.shape}")
        
        # Update input_ids and attention_mask from collate function if available
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to use collate function for dummy data creation: {e}")
        logger.warning("   Falling back to manual creation...")
        import traceback
        traceback.print_exc()
        
        # Fallback to manual creation
        pixel_values = torch.randn(batch_size, num_images, 3, 224, 224)
        height_grid = 14
        width_grid = 14
        num_tokens_per_image = height_grid * width_grid
        image_grid_thw = torch.zeros(batch_size, 3, dtype=torch.long)
        for b in range(batch_size):
            total_tokens = num_tokens_per_image * num_images
            image_grid_thw[b] = torch.tensor([total_tokens, height_grid, width_grid])
    
    # Create input_ids and attention_mask if not created by collate function
    if input_ids is None:
        # Create dummy input_ids with image tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Insert image tokens at random positions
        num_image_tokens = num_images * 256  # Assume 256 tokens per image
        if num_image_tokens < seq_len:
            image_positions = torch.randint(0, seq_len - num_image_tokens, (batch_size,))
            for i, pos in enumerate(image_positions):
                input_ids[i, pos:pos+num_image_tokens] = image_token_id
    
    if attention_mask is None:
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
    
    # Create labels (shifted input_ids for causal LM)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Ignore last token
    
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def setup_model_fast(config_dict, ds_config_dict, device="cuda"):
    """Setup model using train_seqorth.py's setup_model function."""
    model_config = config_dict["model_config"]
    training_config = config_dict.get("training_config", {})
    
    # Set training_config as a module-level variable for train_seqorth.py's setup_model
    # This is needed because setup_model references training_config internally (line 742, 767)
    import training.seqorth_sft.train_seqorth as train_seqorth_module
    
    # Create a mock training_config if not provided, with defaults matching train_seqorth.py usage
    if not training_config:
        training_config = {
            "gradient_checkpointing": False,
            "gradient_checkpointing_kwargs": {"use_reentrant": True},
        }
    
    # Set as module attribute so setup_model can access it
    train_seqorth_module.training_config = training_config
    
    # Use train_seqorth.py's setup_model function
    logger.info("Using train_seqorth.py's setup_model function...")
    setup_result = setup_model(model_config)
    
    # setup_model returns (model, modules_to_save_list)
    if isinstance(setup_result, tuple) and len(setup_result) == 2:
        model, modules_to_save_list = setup_result
    else:
        model = setup_result
        modules_to_save_list = None
    
    return model


def ensure_model_dtype_consistency(model, target_dtype, ds_engine=None):
    """Recursively ensure all model parameters, buffers and submodules use the same dtype."""
    model_to_check = ds_engine.module if ds_engine is not None else model
    
    logger.info(f"Ensuring model consistency for {type(model_to_check).__name__} at {target_dtype}")
    
    # 1. Cast all parameters and buffers that are NOT meta-tensors
    # (In ZeRO-3, some parameters might be meta or sharded)
    for name, param in model_to_check.named_parameters():
        if param is not None and param.device != torch.device("meta"):
            if param.dtype != target_dtype and param.dtype.is_floating_point:
                try:
                    param.data = param.data.to(target_dtype)
                    logger.debug(f"  Cast param {name} to {target_dtype}")
                except Exception as e:
                    logger.warning(f"  Failed to cast param {name}: {e}")
                    
    for name, buffer in model_to_check.named_buffers():
        if buffer is not None and buffer.device != torch.device("meta"):
            if buffer.dtype != target_dtype and buffer.dtype.is_floating_point:
                try:
                    buffer.data = buffer.data.to(target_dtype)
                    logger.debug(f"  Cast buffer {name} to {target_dtype}")
                except Exception as e:
                    logger.warning(f"  Failed to cast buffer {name}: {e}")

    # 2. Deep Search for vision components (Universal Exoskeleton can nest them deeply)
    def _find_and_cast_vision(module, depth=0):
        if depth > 10: return # Prevent infinite loop
        
        # Look for common vision attribute names
        vision_names = ["visual", "vision_tower", "vision_model", "vit"]
        for name in vision_names:
            if hasattr(module, name):
                vis = getattr(module, name)
                if isinstance(vis, torch.nn.Module):
                    try:
                        vis.to(target_dtype)
                        logger.info(f"  ✅ Found and cast vision component '{name}' to {target_dtype}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Failed to cast vision component '{name}': {e}")
        
        # Recurse into children
        for child in module.children():
            _find_and_cast_vision(child, depth + 1)

    _find_and_cast_vision(model_to_check)
    
    # 3. Final safety check for Qwen-specific structures
    if hasattr(model_to_check, "model") and hasattr(model_to_check.model, "visual"):
        try:
            model_to_check.model.visual.to(target_dtype)
            logger.info(f"  ✅ Cast model.model.visual to {target_dtype}")
        except: pass


def test_forward_backward(model, dummy_data, ds_engine=None, config_dict=None, num_steps=5):
    """Test forward and backward pass with explicit dtype handling.
    
    Args:
        model: The model to test
        dummy_data: Dummy input data
        ds_engine: DeepSpeed engine (optional)
        config_dict: Configuration dictionary
        num_steps: Number of training steps to run (default: 5)
    """
    model.train()
    
    # Determine target dtype from config or DeepSpeed config
    target_dtype = None
    if config_dict:
        training_config = config_dict.get("training_config", {})
        if training_config.get("bf16", False):
            target_dtype = torch.bfloat16
        elif training_config.get("fp16", False):
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32
    
    # If not from config, try to get from model or DeepSpeed config
    if target_dtype is None:
        if ds_engine is not None:
            # Check DeepSpeed config
            if hasattr(ds_engine, 'config'):
                ds_config = ds_engine.config
                if ds_config.get("bf16", {}).get("enabled", False):
                    target_dtype = torch.bfloat16
                elif ds_config.get("fp16", {}).get("enabled", False):
                    target_dtype = torch.float16
                else:
                    target_dtype = torch.float32
        
        # Fallback: try to get from model parameters
        if target_dtype is None:
            try:
                # For ZeRO-3, parameters might be meta or sharded, but we can usually get dtype
                sample_param = next(model.parameters())
                target_dtype = sample_param.dtype
            except:
                target_dtype = torch.bfloat16  # Default fallback
    
    logger.info(f"Using target dtype: {target_dtype}")
    
    # CRITICAL: Ensure all model components use the same dtype
    # This is especially important for DeepSpeed Zero Stage 3 where parameters are sharded
    ensure_model_dtype_consistency(model, target_dtype, ds_engine)
    
    # Prepare inputs
    try:
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    except:
        # Fallback for ZeRO-3 where parameters might be sharded
        device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}" if dist.is_initialized() else "cuda:0")
    
    # Move tensors to device and cast to target_dtype
    # CRITICAL: For DeepSpeed Zero Stage 3, all float inputs must match model dtype
    # to avoid "BFloat16 and Half" mismatch in linear operations
    input_ids = dummy_data["input_ids"].to(device)
    labels = dummy_data["labels"].to(device)
    attention_mask = dummy_data["attention_mask"].to(device)
    
    # pixel_values and other float tensors must match model dtype
    # CRITICAL: Ensure pixel_values are in the correct dtype to avoid BFloat16/Half mismatch
    # DeepSpeed Zero Stage 3's linear wrapper requires input and weight dtypes to match exactly
    pixel_values = dummy_data["pixel_values"]
    if pixel_values.dtype.is_floating_point:
        # Always cast floating point tensors to target_dtype
        pixel_values = pixel_values.to(device, dtype=target_dtype)
        logger.debug(f"Cast pixel_values from {dummy_data['pixel_values'].dtype} to {target_dtype}")
    else:
        pixel_values = pixel_values.to(device)
    
    image_grid_thw = dummy_data.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    
    # Verify consistency one last time before forward
    if dist.is_initialized():
        dist.barrier()
    
    # Run multiple training steps
    losses = []
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    logger.info(f"Rank {rank}: Starting {num_steps} training steps...")
    
    try:
        for step in range(1, num_steps + 1):
            logger.info(f"Rank {rank}: ========== Step {step}/{num_steps} ==========")
            
            # Forward pass
            logger.info(f"Rank {rank}: Starting forward pass (step {step})...")
            if ds_engine is not None:
                # CRITICAL (by-design behavior):
                # If you are using DeepSpeed's built-in bf16/fp16, you should NOT enable torch.autocast externally.
                # Mixing them can lead to partial casts and ZeRO-3 Linear seeing mismatched dtypes
                # (e.g. bias BF16 vs weight FP16) -> crash.
                if torch.is_autocast_enabled():
                    raise RuntimeError(
                        "torch.autocast is enabled while using DeepSpeed engine. "
                        "Disable torch autocast and rely on DeepSpeed bf16/fp16, or configure DeepSpeed `torch_autocast`."
                    )

                outputs = ds_engine(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                    attention_mask=attention_mask,
                )
            else:
                # Non-DeepSpeed path: use torch autocast for bf16/fp16 to match target_dtype.
                autocast_type = target_dtype if target_dtype in [torch.bfloat16, torch.float16] else None
                enabled = target_dtype in [torch.bfloat16, torch.float16]
                if enabled and autocast_type is not None:
                    with torch.amp.autocast(device_type="cuda", dtype=autocast_type, enabled=True):
                        outputs = model(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            labels=labels,
                            attention_mask=attention_mask,
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            loss_value = loss.item()
            losses.append(loss_value)
            
            if dist.is_initialized(): 
                dist.barrier()
            logger.info(f"Rank {rank}: Forward pass completed (step {step}). Loss: {loss_value:.4f}")
            
            # Backward pass
            logger.info(f"Rank {rank}: Starting backward pass (step {step})...")
            if ds_engine is not None:
                ds_engine.backward(loss)
                ds_engine.step()
            else:
                loss.backward()
                # For non-DeepSpeed, manually zero gradients after backward
                if step < num_steps:  # Don't zero on last step
                    model.zero_grad()
            
            if dist.is_initialized(): 
                dist.barrier()
            logger.info(f"Rank {rank}: Backward pass completed (step {step}).")
        
        avg_loss = sum(losses) / len(losses)
        logger.info(f"Rank {rank}: ========== Completed {num_steps} steps ==========")
        logger.info(f"Rank {rank}: Average loss: {avg_loss:.4f} (Losses: {[f'{l:.4f}' for l in losses]})")
        
        return avg_loss
    except Exception as e:
        logger.error(f"❌ Rank {rank} failed in step {step if 'step' in locals() else 'unknown'}: {e}")
        import traceback
        traceback.print_exc()
        raise e


def main():
    """Main test function."""
    import argparse
    
    # Parse command line arguments (same as train_seqorth.py)
    parser = argparse.ArgumentParser(description="Fast DeepSpeed Forward/Backward Test")
    # DeepSpeed launcher injects --local_rank; accept it for compatibility.
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="Local rank passed by DeepSpeed launcher (ignored; we read LOCAL_RANK env as source of truth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/seqorth_sft/config/seqorth_qwen_config.json",
        help="Path to training configuration JSON file"
    )
    args = parser.parse_args()
    
    # Paths (repo root = tests/training_tests -> tests -> repo root)
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = Path(args.config) if not Path(args.config).is_absolute() else Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    # Load config to get deepspeed_config path
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Get deepspeed_config path from model_config or training_config
    deepspeed_config_path_str = None
    if "model_config" in config_dict and "deepspeed_config" in config_dict["model_config"]:
        deepspeed_config_path_str = config_dict["model_config"]["deepspeed_config"]
    elif "training_config" in config_dict and "deepspeed" in config_dict["training_config"]:
        deepspeed_config_path_str = config_dict["training_config"]["deepspeed"]
    
    if deepspeed_config_path_str:
        deepspeed_config_path = Path(deepspeed_config_path_str)
        if not deepspeed_config_path.is_absolute():
            deepspeed_config_path = project_root / deepspeed_config_path
    else:
        # Default fallback
        deepspeed_config_path = project_root / "training" / "seqorth_sft" / "config" / "deepspeed_auto.json"
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    if not deepspeed_config_path.exists():
        logger.error(f"DeepSpeed config file not found: {deepspeed_config_path}")
        return
    
    # Setup distributed
    is_distributed = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    # Load configs (config_dict already loaded above)
    logger.info("Loading configurations...")
    batch_size = config_dict.get("training_config", {}).get("per_device_train_batch_size", 1)
    _, ds_config_dict = load_configs(
        str(config_path), 
        str(deepspeed_config_path),
        batch_size=batch_size
    )
    
    # Read TP size from Accelerate config (parallelism_config)
    tp_size = 1
    try:
        import yaml
        accelerate_config_path = project_root / "training" / "seqorth_sft" / "config" / "accelerate.yaml"
        if accelerate_config_path.exists():
            with open(accelerate_config_path, 'r') as f:
                accelerate_config = yaml.safe_load(f)
                parallelism_config = accelerate_config.get("parallelism_config", {})
                tp_size = parallelism_config.get("tp_size", 1)
                logger.info(f"📊 Read TP size from Accelerate config: {tp_size}")
        else:
            logger.warning(f"⚠️ Accelerate config not found at {accelerate_config_path}, using TP size 1")
    except Exception as e:
        logger.warning(f"⚠️ Failed to read Accelerate config for TP size: {e}, using TP size 1")
    
    # Apply TP size to DeepSpeed config if TP > 1
    if tp_size > 1 and ds_config_dict:
        ds_config_dict["tensor_model_parallel_size"] = tp_size
        logger.info(f"✅ Applied TP size {tp_size} to DeepSpeed config")

    # CRITICAL:
    # If DeepSpeed is driving mixed precision (bf16/fp16 section), do not allow global torch autocast.
    # The warning you saw ("torch.autocast is enabled outside DeepSpeed but disabled within the engine")
    # is exactly the situation that can create BF16/Half mismatches under ZeRO-3 Linear.
    if ds_config_dict:
        ds_mp_enabled = bool(ds_config_dict.get("bf16", {}).get("enabled", False) or ds_config_dict.get("fp16", {}).get("enabled", False))
        if ds_mp_enabled:
            torch.set_autocast_enabled(False)
            logger.info("DeepSpeed mixed precision enabled -> torch.set_autocast_enabled(False)")
    
    # Save modified DeepSpeed config to a temporary file
    # This ensures setup_model (which reads from disk) uses the SAME config as deepspeed.initialize
    import tempfile
    temp_ds_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(ds_config_dict, temp_ds_config)
    temp_ds_config_path = temp_ds_config.name
    temp_ds_config.close()
    
    # Override the config path in config_dict
    config_dict["model_config"]["deepspeed_config"] = temp_ds_config_path
    logger.info(f"Using temporary DeepSpeed config: {temp_ds_config_path}")
    
    # Setup model first (needed to get vocab_size for dummy data)
    logger.info("Setting up model with DeepSpeed ZeRO-3...")
    try:
        model = setup_model_fast(config_dict, ds_config_dict, device=device)
        
        # Determine and apply global dtype from config or DeepSpeed config
        target_dtype = None
        training_config = config_dict.get("training_config", {})
        if training_config.get("bf16", False):
            target_dtype = torch.bfloat16
        elif training_config.get("fp16", False):
            target_dtype = torch.float16
        else:
            # Check DeepSpeed config
            if ds_config_dict:
                if ds_config_dict.get("bf16", {}).get("enabled", False):
                    target_dtype = torch.bfloat16
                elif ds_config_dict.get("fp16", {}).get("enabled", False):
                    target_dtype = torch.float16
                else:
                    target_dtype = torch.float32
            else:
                target_dtype = torch.float32
        
        logger.info(f"Force casting model to {target_dtype} for consistency...")
        # CRITICAL: Cast model before DeepSpeed initialization
        # After DeepSpeed init, parameters are sharded and cannot be directly cast
        model.to(target_dtype)
        
        # Ensure vision tower is also cast to target dtype
        if hasattr(model, 'visual'):
            try:
                model.visual.to(target_dtype)
                logger.info(f"✅ Vision tower cast to {target_dtype} before DeepSpeed init")
            except Exception as e:
                logger.warning(f"Could not cast vision tower before DeepSpeed init: {e}")
        
        logger.info("Model setup completed.")
        
        # CRITICAL: Clear GPU memory before DeepSpeed initialization
        # DeepSpeed ZeRO-3 will handle sharding, so we can free up memory here
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        if dist.is_initialized():
            dist.barrier()
        logger.info("✅ Cleared GPU memory before DeepSpeed initialization")
        
    except Exception as e:
        logger.error(f"Model setup failed: {e}", exc_info=True)
        if Path(temp_ds_config_path).exists():
            os.unlink(temp_ds_config_path)
        return
    
    # Create dummy data (after model is loaded to get correct vocab_size)
    logger.info("Creating dummy data...")
    dummy_data = create_dummy_data(
        config_dict,
        model=model,
        batch_size=batch_size,
        seq_len=128,
        num_images=1
    )
    
    # Cleanup temp config file
    if Path(temp_ds_config_path).exists():
        os.unlink(temp_ds_config_path)
    
    # Verify dummy data consistency across ranks
    if not verify_rank_consistency(dummy_data, logger):
        raise RuntimeError("Rank desync in dummy data detected! All ranks must have identical inputs for ZeRO-3 MoE testing.")
    
    # Check MoE Patch application
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts
    if hasattr(Qwen3VLMoeTextExperts, "forward"):
        # Check if it's the patched version (we can't easily check identity of local functions, 
        # but we can log that we are using the class that should be patched)
        logger.info(f"MoE Expert class detected: {Qwen3VLMoeTextExperts.__name__}")
    
    # Initialize DeepSpeed engine
    model_engine = None
    if ds_config_dict:
        logger.info("Initializing DeepSpeed engine...")
        try:
            # ZeRO-Offload (NVMe) requires DeepSpeed to manage the optimizer for efficiency
            zero_offload_optimizer = (
                ds_config_dict.get("zero_optimization", {}).get("offload_optimizer", {}).get("device") is not None
            )
            
            if zero_offload_optimizer:
                if "optimizer" not in ds_config_dict:
                    logger.info("Adding optimizer to DeepSpeed config for ZeRO-Offload (NVMe)...")
                    ds_config_dict["optimizer"] = {
                        "type": "AdamW",
                        "params": {
                            "lr": 1e-5,
                            "weight_decay": 0.01,
                            "betas": [0.9, 0.999],
                            "eps": 1e-8
                        }
                    }
                optimizer_to_pass = None
                lr_scheduler_to_pass = None
            else:
                # Fallback for non-offload
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
                optimizer_to_pass = optimizer
                lr_scheduler_to_pass = None

            # Fix engine_timers issue
            ds_config_dict["wall_clock_breakdown"] = True
            
            if dist.is_initialized():
                dist.barrier()
                
            logger.info("🚀 Initializing DeepSpeed engine (ZeRO-3 + NVMe Offload)...")
            logger.info("   Note: This can take 10-20 minutes for a 30B model as parameters are sharded to NVMe.")
            
            model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer_to_pass,
                lr_scheduler=lr_scheduler_to_pass,
                config=ds_config_dict,
            )
            
            logger.info("✅ DeepSpeed engine initialized successfully.")
            
            # CRITICAL: After DeepSpeed initialization, ensure dtype consistency
            # DeepSpeed Zero Stage 3 may have sharded parameters, but buffers and inputs must match
            target_dtype = torch.bfloat16 if ds_config_dict.get("bf16", {}).get("enabled", False) else (
                torch.float16 if ds_config_dict.get("fp16", {}).get("enabled", False) else torch.float32
            )
            logger.info(f"Ensuring dtype consistency after DeepSpeed init: {target_dtype}")
            ensure_model_dtype_consistency(model, target_dtype, ds_engine=model_engine)
        except Exception as e:
            logger.error(f"DeepSpeed initialization failed: {e}", exc_info=True)
            logger.error("Cannot continue without DeepSpeed engine - model is in ZeRO-3 sharded state")
            logger.error("Model parameters are sharded and cannot be used without DeepSpeed engine")
            raise RuntimeError(f"DeepSpeed engine initialization failed: {e}. Cannot proceed without DeepSpeed engine.")
    
    # Run test
    num_steps = 5
    logger.info("=" * 80)
    logger.info(f"Starting forward/backward test with {num_steps} steps...")
    logger.info("=" * 80)
    
    try:
        loss = test_forward_backward(
            model_engine if model_engine else model, 
            dummy_data, 
            ds_engine=model_engine,
            config_dict=config_dict,
            num_steps=num_steps
        )
        logger.info("=" * 80)
        logger.info(f"✅ Test completed successfully! Average loss over {num_steps} steps: {loss:.4f}")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise
    
    # Cleanup
    if is_distributed:
        try:
            dist.barrier()  # Ensure all ranks finish before destroying
            dist.destroy_process_group()
        except Exception as e:
            logger.warning(f"Error during process group cleanup: {e}")
    
    # Additional cleanup for grpc resources
    cleanup_grpc()


if __name__ == "__main__":
    main()

