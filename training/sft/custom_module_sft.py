#!/usr/bin/env python3
"""
SPECTRA MoE SFT Training Script using Config File
"""

import os
import sys
import json
import torch
import traceback
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, Any
from torchinfo import summary
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM
)
from transformers import logging as transformers_logging

from transformers.trainer_utils import set_seed
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import TaskType
import wandb

# Add parent directory to path to import custom modules
# Expect PYTHONPATH to include repo root

# Import custom modules  
from models.spectra import upcycle_model_to_moe, SPECTRARouter, SPECTRABlock
from models.g3moe_model import G3MoEMLP
from data.base_model_sft_dataset import get_dataset, create_multimodal_collate_fn
from data.simple_sft_dataset import get_simple_sft_dataset, create_simple_collate_fn, smoltalk_dataset, orca_mini_dataset, validate_image_data

from core.training_utils.utils import format_parameters, load_config, setup_deepspeed_environment
from core.optimizers.custom_optimizers import get_custom_optimizer
from core.optimizers.deepspeed_optimizer_registry import register_custom_optimizers
from eval.callbacks import ModelEvalCallback
from eval.ifeval_callback import IFEvalCallback
from eval.moe_monitoring_callback import create_moe_callback_for_transformers

# Register custom optimizers with DeepSpeed
register_custom_optimizers()

transformers_logging.enable_progress_bar()
transformers_logging.set_verbosity_warning()

# Setup comprehensive logging system
def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Setup comprehensive logging system for training monitoring"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"training_detailed_{timestamp}.log"),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # File handler for error logs
    error_handler = logging.FileHandler(
        os.path.join(log_dir, f"training_errors_{timestamp}.log"),
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_logging()

def log_gpu_memory(logger, stage: str, device: int = 0):
    """Log detailed GPU memory information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3    # GB
        
        logger.info(f"🔧 GPU Memory [{stage}] - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        logger.debug(f"🔧 GPU Memory [{stage}] - Max Allocated: {max_allocated:.2f}GB, Max Reserved: {max_reserved:.2f}GB")
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'max_reserved': max_reserved
        }
    return None

def log_training_progress(logger, trainer, step: int = None, epoch: float = None, loss: float = None):
    """Log detailed training progress information"""
    if hasattr(trainer, 'state') and trainer.state is not None:
        state = trainer.state
        current_step = step or state.global_step
        current_epoch = epoch or state.epoch
        current_loss = loss or getattr(state, 'log_history', [{}])[-1].get('train_loss', 'N/A')
        
        logger.info(f"📊 Training Progress - Step: {current_step}, Epoch: {current_epoch:.3f}, Loss: {current_loss}")
        
        # Log learning rate if available
        if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None:
            lr = trainer.lr_scheduler.get_last_lr()[0] if hasattr(trainer.lr_scheduler, 'get_last_lr') else 'N/A'
            logger.debug(f"📊 Learning Rate: {lr}")
        
        # Log gradient norm if available
        if hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
            if hasattr(trainer.accelerator, 'unwrap_model'):
                model = trainer.accelerator.unwrap_model(trainer.model)
                total_norm = 0
                param_count = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                if param_count > 0:
                    total_norm = total_norm ** (1. / 2)
                    logger.debug(f"📊 Gradient Norm: {total_norm:.6f}")

def log_error_context(logger, error: Exception, context: str = ""):
    """Log detailed error context with system state"""
    logger.error(f"❌ Error in {context}: {str(error)}")
    logger.error(f"❌ Error type: {type(error).__name__}")
    
    # Log traceback
    logger.error(f"❌ Traceback:\n{traceback.format_exc()}")
    
    # Log GPU memory state
    if torch.cuda.is_available():
        memory_info = log_gpu_memory(logger, "ERROR")
        if memory_info:
            logger.error(f"❌ GPU Memory at error - Allocated: {memory_info['allocated']:.2f}GB, Reserved: {memory_info['reserved']:.2f}GB")
    
    # Log system state
    logger.error(f"❌ System state - CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logger.error(f"❌ Current device: {torch.cuda.current_device()}, Device name: {torch.cuda.get_device_name()}")

def load_config(config_path: str):
    """간단한 config 로더"""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_deepspeed_environment():
    """Setup environment variables for DeepSpeed optimization"""
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if "DEEPSPEED_ZERO_INIT" not in os.environ:
        os.environ["DEEPSPEED_ZERO_INIT"] = "1"
    # Ensure global AMP default uses BF16 under CUDA
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch.set_autocast_gpu_dtype(torch.bfloat16)
    except Exception as _:
        pass
    print("DeepSpeed environment variables set")


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection with detailed logging"""
    import gc
    logger.info("🧹 Starting GPU memory cleanup...")
    
    # Log memory before cleanup
    memory_before = log_gpu_memory(logger, "BEFORE_CLEANUP")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("🧹 CUDA cache cleared and synchronized")
    
    # Force garbage collection
    collected = gc.collect()
    logger.debug(f"🧹 Garbage collection freed {collected} objects")
    
    # Log memory after cleanup
    memory_after = log_gpu_memory(logger, "AFTER_CLEANUP")
    
    if memory_before and memory_after:
        freed_allocated = memory_before['allocated'] - memory_after['allocated']
        freed_reserved = memory_before['reserved'] - memory_after['reserved']
        logger.info(f"🧹 Memory cleanup completed - Freed: {freed_allocated:.2f}GB allocated, {freed_reserved:.2f}GB reserved")
    else:
        logger.info("🧹 Memory cleanup completed")


def eval_with_memory_optimization(trainer, original_eval_fn, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    """Memory-optimized evaluation function with detailed logging"""
    logger.info("🔧 Starting memory-optimized evaluation...")
    
    # Log evaluation context
    if hasattr(trainer, 'state') and trainer.state is not None:
        logger.info(f"🔧 Evaluation context - Step: {trainer.state.global_step}, Epoch: {trainer.state.epoch:.3f}")
    
    # Log memory before evaluation
    memory_before = log_gpu_memory(logger, "BEFORE_EVAL")
    
    # GPU 메모리 정리
    clear_gpu_memory()
    
    # 모델을 eval 모드로 설정하고 메모리 최적화
    logger.debug("🔧 Setting model to eval mode...")
    trainer.model.eval()
    
    # eval 시에는 gradient checkpointing 비활성화
    original_gc = trainer.args.gradient_checkpointing
    trainer.args.gradient_checkpointing = False
    logger.debug(f"🔧 Disabled gradient checkpointing for evaluation (was: {original_gc})")
    
    try:
        logger.info("🔧 Starting evaluation with torch.no_grad()...")
        start_time = time.time()
        
        with torch.no_grad():
            # 원래 evaluate 함수 호출 (무한 재귀 방지)
            eval_results = original_eval_fn(
                eval_dataset=eval_dataset, 
                ignore_keys=ignore_keys, 
                metric_key_prefix=metric_key_prefix
            )
        
        eval_time = time.time() - start_time
        logger.info(f"🔧 Evaluation completed in {eval_time:.2f} seconds")
        
        # Log evaluation results
        if eval_results:
            logger.info(f"🔧 Evaluation results: {eval_results}")
        
        # Log memory after evaluation
        memory_after = log_gpu_memory(logger, "AFTER_EVAL")
        
        # 결과 반환
        return eval_results
        
    except Exception as e:
        logger.error(f"❌ Error during evaluation: {str(e)}")
        log_error_context(logger, e, "memory_optimized_evaluation")
        raise e
        
    finally:
        # 원래 설정 복원
        logger.debug(f"🔧 Restoring gradient checkpointing to: {original_gc}")
        trainer.args.gradient_checkpointing = original_gc
        clear_gpu_memory()


def setup_model_and_tokenizer(model_config: Dict[str, Any]):
    """Setup SPECTRA MoE model and tokenizer with detailed logging"""
    logger.info("🚀 Starting model and tokenizer setup...")
    
    # NOTE: Delay DeepSpeed env setup until AFTER model load to avoid HF ZeRO-3 init slow path
    logger.info("🔧 Setting up DeepSpeed environment...")
    setup_deepspeed_environment()
    
    # Load tokenizer - 안정적인 로딩 로직
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
        
        # AutoProcessor인 경우 tokenizer 속성에 설정
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.tokenizer.chat_template = chat_template
            print("  ✅ 채팅 템플릿을 tokenizer.tokenizer에 설정")
        else:
            tokenizer.chat_template = chat_template
            print("  ✅ 채팅 템플릿을 tokenizer에 설정")
        
        print(f"  - 템플릿 길이: {len(chat_template)}")
    except Exception as e:
        print(f"  ⚠️ 채팅 템플릿 설정 실패: {e}")
        print("  - 기본 템플릿으로 계속 진행")
    
    # Set padding side for multimodal models
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer.tokenizer.padding_side = "right"
        print("  ✅ tokenizer.tokenizer.padding_side = 'right' 설정")
    else:
        tokenizer.padding_side = "right"
        print("  ✅ tokenizer.padding_side = 'right' 설정")

    # Ensure tokenizer has pad token
    if not hasattr(tokenizer, 'pad_token'):
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.pad_token = tokenizer.tokenizer.pad_token if tokenizer.tokenizer.pad_token is not None else tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if not hasattr(tokenizer, 'convert_tokens_to_ids'):
        tokenizer.convert_tokens_to_ids = tokenizer.tokenizer.convert_tokens_to_ids

    # Prefer config value; default to eager
    attn_from_cfg = (model_config.get("g3moe_params") or {}).get("attn_implementation")
    if attn_from_cfg in {"eager", "sdpa", "flash_attention_2"}:
        attn_implementation = attn_from_cfg
    else:
        attn_implementation = "eager"

    # Load base model configuration
    print("Loading base model configuration...")
    base_config = AutoConfig.from_pretrained(
        model_config["model_name_or_path"],
        trust_remote_code=model_config["trust_remote_code"]
    )
    
    # SPECTRA MoE configuration parameters from config file
    g3moe_params = model_config.get("g3moe_params", {})
    
    # Extract model dimensions from config
    if hasattr(base_config, 'text_config') and base_config.text_config is not None:
        # Multi-modal model with text_config
        hidden_size = base_config.text_config.hidden_size
        intermediate_size = base_config.text_config.intermediate_size
        num_hidden_layers = base_config.text_config.num_hidden_layers
        hidden_activation = getattr(base_config.text_config, 'hidden_act', 'silu')
    else:
        # Direct text model config
        hidden_size = base_config.hidden_size
        intermediate_size = base_config.intermediate_size
        num_hidden_layers = base_config.num_hidden_layers
        hidden_activation = getattr(base_config, 'hidden_act', 'silu')
    
    print("SPECTRA MoE configuration:")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Intermediate size: {intermediate_size}")
    print(f"  - Num hidden layers: {num_hidden_layers}")
    
    # Load model - use different device_map strategy based on DeepSpeed usage
    device_map = None
    if model_config.get("deepspeed_config"):
        # With DeepSpeed, let DeepSpeed handle device placement
        device_map = None
        print("Using DeepSpeed - letting DeepSpeed handle device placement")
    elif torch.cuda.device_count() > 1:
        # Without DeepSpeed, use auto device mapping for multi-GPU
        device_map = "auto"
        print(f"Using auto device mapping for {torch.cuda.device_count()} GPUs")
    
    # Load base model first, then upcycle to SPECTRA MoE
    logger.info("🤖 Loading base model...")
    logger.info(f"🤖 Model path: {model_config['model_name_or_path']}")
    logger.info(f"🤖 Device map: {device_map}")
    logger.info(f"🤖 Attention implementation: {attn_implementation}")
    
    # Log memory before model loading
    memory_before = log_gpu_memory(logger, "BEFORE_MODEL_LOAD")
    
    try:
        start_time = time.time()
        # Load base model (can be any HuggingFace model)
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name_or_path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=model_config["trust_remote_code"],
            device_map=device_map,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            use_cache=False,
            attn_implementation=attn_implementation
        )
        load_time = time.time() - start_time
        logger.info(f"✅ Base model loaded successfully in {load_time:.2f} seconds")
        
        # Log memory after base model loading
        memory_after_base = log_gpu_memory(logger, "AFTER_BASE_MODEL_LOAD")
        
        total_params_before = sum(p.numel() for p in model.parameters())
        logger.info(f"  - Total parameters (before upcycling): {format_parameters(total_params_before)}")
        
        # Prepare SPECTRA MoE configuration
        moe_config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": g3moe_params.get("n_routed_experts", 8),
            "num_experts_per_tok": g3moe_params.get("num_experts_per_tok", 2),
            "router_dim": 128,  # Default router dimension
            "n_shared_experts": g3moe_params.get("n_shared_experts", 1),
            "first_k_dense_replace": g3moe_params.get("first_k_dense_replace", 0),
            "router_jitter_noise": g3moe_params.get("router_jitter_noise", 0.01),
            "input_jitter_noise": g3moe_params.get("input_jitter_noise", 0.0),
            "freeze_shared_experts": g3moe_params.get("freeze_shared_experts", True),
            "balancing_strength": 0.01,
            "ema_alpha": 0.99,
            "hidden_activation": hidden_activation,
        }
        
        logger.info("🔄 Upcycling model to SPECTRA MoE...")
        logger.info(f"  - Num experts: {moe_config['num_experts']}")
        logger.info(f"  - Num experts per token: {moe_config['num_experts_per_tok']}")
        logger.info(f"  - Shared experts: {moe_config['n_shared_experts']}")
        logger.info(f"  - First k dense layers: {moe_config['first_k_dense_replace']}")
        
        # Upcycle to SPECTRA MoE
        upcycle_start = time.time()
        model = upcycle_model_to_moe(
            model,
            moe_config,
            expert_module_class=G3MoEMLP,
            layer_start_idx=moe_config["first_k_dense_replace"],
            verbose=True
        )
        upcycle_time = time.time() - upcycle_start
        logger.info(f"✅ SPECTRA MoE upcycling completed in {upcycle_time:.2f} seconds")
        
        # Log memory after upcycling
        memory_after = log_gpu_memory(logger, "AFTER_MOE_UPcycling")
        
        total_params_after = sum(p.numel() for p in model.parameters())
        logger.info(f"  - Total parameters (after upcycling): {format_parameters(total_params_after)}")
        logger.info(f"  - Parameter increase: {format_parameters(total_params_after - total_params_before)}")
        
        # Log model device placement
        if hasattr(model, 'device'):
            logger.info(f"  - Model device: {model.device}")
        elif hasattr(model, 'hf_device_map'):
            logger.info(f"  - Model device map: {model.hf_device_map}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load/upcycle model: {str(e)}")
        log_error_context(logger, e, "model_loading")
        raise e

    # Setup LoRA if requested
    if model_config["use_lora"]:
        # G3MoERouter/SPECTRARouter는 PEFT에서 지원하지 않으므로 target_modules에서 제외
        # Router는 PEFT 적용 후 수동으로 trainable로 설정
        lora_config = LoraConfig(
            r=model_config["lora_r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            target_modules=[
                # "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                # "router", "routing_temperature" 제외 - PEFT 미지원
                "rnn.weight_ih_l0", "rnn.weight_hh_l0"
            ],
            # modules_to_save에서도 router 제외 (PEFT가 처리할 수 없음)
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,  # 훈련 모드 명시
            fan_in_fan_out=False,  # LoRA 호환성 향상
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()
        
        # LoRA 어댑터 설정
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
        
        # Router 모듈을 찾아서 trainable로 설정 (PEFT 적용 후)
        from models.g3moe_model import G3MoERouter
        from models.spectra import SPECTRARouter
        router_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (G3MoERouter, SPECTRARouter)):
                for p in module.parameters(recurse=True):
                    p.requires_grad_(True)
                router_count += 1
                logger.info(f"✓ Router module '{name}' set to trainable (not LoRA-wrapped)")
        
        if router_count > 0:
            logger.info(f"✓ {router_count} router module(s) set to fully trainable")
        else:
            logger.warning("⚠️ No router modules found - router may not be trainable")
        # DDP 정적 그래프 비활성화: MoE 라우팅/LoRA로 스텝마다 활성 파라미터가 달라질 수 있으므로 동적 그래프 허용
        if hasattr(model, '_set_static_graph'):
            model._set_static_graph(True)
        # Ensure all parameters incl. LoRA adapters are bfloat16 for consistency
        try:
            model.to(torch.bfloat16)
            for name, param in model.named_parameters():
                if param.requires_grad and param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)
            print("✓ Parameters cast to bfloat16")
        except Exception as cast_e:
            print(f"⚠️ BF16 cast warning: {cast_e}")
        print("✓ LoRA 적용")
        
    return model, tokenizer


def setup_dataset(data_config: Dict[str, Any], tokenizer):
    """Setup training dataset"""    
    dataset_name = data_config.get("dataset_name", "HuggingFaceTB/smoltalk")
    max_samples = data_config.get("max_samples", 100000)
    max_seq_length = data_config.get("max_seq_length", 131072) or 131072
    test_size = data_config.get("test_size", 0.1)
    
    print(f"Loading simple SFT dataset: {dataset_name}")
    print(f"  - Max samples: {max_samples}")
    print(f"  - Max sequence length: {max_seq_length}")
    print(f"  - Test size: {test_size}")
    print(f"  - 토크나이저 타입: {type(tokenizer)}")
    print(f"  - 토크나이저에 chat_template 있음: {hasattr(tokenizer, 'chat_template')}")
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print(f"  - chat_template 길이: {len(str(tokenizer.chat_template))}")
    else:
        print(f"  - ⚠️ chat_template이 설정되지 않음!")
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "sft", "chat_template.txt"), "r") as f:
        chat_template = f.read()
        
        # AutoProcessor인 경우 tokenizer 속성에 설정
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer.tokenizer.chat_template = chat_template
            print("  ✅ 채팅 템플릿을 tokenizer.tokenizer에 설정")
        
        tokenizer.chat_template = chat_template
        print("  ✅ 채팅 템플릿을 tokenizer에 설정")
    
    # print(f"Loading dataset: {data_config['dataset_name']}")
    try:
        # 간단한 데이터셋 로더 사용
        if "smoltalk" in dataset_name.lower() or "orca" in dataset_name.lower() or "llava" in dataset_name.lower():
            print(f"일반 데이터셋 로더 시도: {dataset_name}")
            dataset = get_simple_sft_dataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                max_samples=max_samples,
                test_size=test_size,
                use_streaming=True
            )
            # 이미지 중첩 리스트 문제 해결을 위한 커스텀 data collator 사용
            collate_fn = create_simple_collate_fn(tokenizer, max_length=max_seq_length)
        else:
            # open_m_3 데이터셋 로더 시도
            dataset = get_dataset(
                tokenizer=tokenizer,
                dataset_name=data_config["dataset_name"],
                max_length=data_config["max_seq_length"],
                test_size=data_config["test_size"],
                text_only=data_config["text_only"],
                streaming=data_config["streaming"]
            )
            collate_fn = create_multimodal_collate_fn(tokenizer)
        
        print(f"Dataset loaded:")
        for split, data in dataset.items():
            print(f"  {split}: {data.info.dataset_size if bool(data_config['streaming']) else len(data)} examples")
        
        # 빈 데이터셋 체크
        if data_config.get("streaming", False):
            if dataset.get("train", []).info.dataset_size == 0:
                raise ValueError("훈련 데이터셋이 비어있습니다!")

        return dataset, collate_fn
        
    except Exception as e:
        print(f"❌ 데이터셋 로딩 실패: {e}")
        assert False, "데이터셋 로딩 실패"
        print("🔄 대안 데이터셋으로 재시도 (SmolTalk)")
        try:
            dataset = smoltalk_dataset(tokenizer, max_samples=max_samples)
            print(f"대안 데이터셋 로드 성공:")
            for split, data in dataset.items():
                print(f"  {split}: {len(data)} examples")
            return dataset
        except Exception as e2:
            print(f"❌ 대안 데이터셋도 실패: {e2}")
            raise RuntimeError(f"모든 데이터셋 로딩 시도가 실패했습니다: {e2}")


def create_training_args(
    training_config: Dict[str, Any], 
    deepspeed_config: str | None = None 
) -> SFTConfig:
    """Create SFTConfig from training configuration"""
    
    # Create SFTConfig with all parameters
    training_args = SFTConfig(
        **training_config,
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    
    # Add DeepSpeed config if provided
    if deepspeed_config:
        import os, json
        ds_cfg_path_abs = os.path.abspath(deepspeed_config)
        training_args.deepspeed = ds_cfg_path_abs
        print(f"DeepSpeed config set: {ds_cfg_path_abs}")
        # Validate that CPU offload is disabled as required
        try:
            with open(ds_cfg_path_abs, "r") as f:
                ds_cfg = json.load(f)
            zero = ds_cfg.get("zero_optimization", {})
            off_opt = (zero.get("offload_optimizer") or {}).get("device", "none").lower()
            off_param = (zero.get("offload_param") or {}).get("device", "none").lower()
            print(f"DeepSpeed zero stage: {zero.get('stage')}")
            print(f"DeepSpeed offload_optimizer.device: {off_opt}")
            print(f"DeepSpeed offload_param.device: {off_param}")
            # assert off_opt in {"none", None, ""} and off_param in {"none", None, ""}, (
            #     "DeepSpeed CPU offload detected in config but must be disabled (device='none')."
            # )
            # Workaround: ZeRO-3 + gradient checkpointing can trigger duplicate ds_id assertion
            try:
                zero_stage = int(zero.get("stage", 0) or 0)
            except Exception:
                zero_stage = 0
            # if zero_stage == 3 and getattr(training_args, "gradient_checkpointing", False):
            #     print("⚠️ Detected ZeRO-3 with gradient checkpointing enabled. Disabling to avoid ds_id assertion.")
            #     training_args.gradient_checkpointing = False
        except Exception as e:
            print(f"⚠️ DeepSpeed config validation warning: {e}")
    
    return training_args


def main(
    model_config: Dict[str, Any], 
    data_config: Dict[str, Any], 
    training_config: Dict[str, Any]
):
    register_custom_optimizers()
    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_config)
    
    # Setup dataset
    print("Setting up dataset...")
    dataset, collate_fn = setup_dataset(data_config, tokenizer)
    
    # Create training arguments
    training_args = create_training_args(
        training_config, 
        model_config.get("deepspeed_config")
    )
    
    # Optionally build a custom optimizer (e.g., Muon) prior to DeepSpeed init
    custom_optimizer = None
    try:
        ds_cfg_path = model_config.get("deepspeed_config")
        if ds_cfg_path:
            with open(ds_cfg_path, "r") as f:
                ds_cfg = json.load(f)
            # Prefer explicit custom optimizer block
            custom_opt_section = ds_cfg.get("custom_optimizer")
            from core.optimizers.deepspeed_optimizer_registry import create_optimizer_from_config
            if custom_opt_section:
                trainable_params = (p for p in model.parameters() if p.requires_grad)
                custom_optimizer = create_optimizer_from_config(custom_opt_section, trainable_params)
                print(f"✓ Using custom optimizer: {custom_opt_section.get('type')}")
            else:
                # Fallback: if optimizer.type is a custom one, build it here
                opt_section = ds_cfg.get("optimizer")
                if opt_section:
                    opt_type = str(opt_section.get("type", "")).lower()
                    if opt_type in {"muon", "muonoptimizer", "lion", "adafactor", "sophia"}:
                        trainable_params = (p for p in model.parameters() if p.requires_grad)
                        custom_optimizer = create_optimizer_from_config(opt_section, trainable_params)
                        print(f"✓ Using custom optimizer from optimizer block: {opt_section.get('type')}")
    except Exception as opt_e:
        print(f"⚠️ Custom optimizer setup skipped: {opt_e}")

    # Setup trainer
    print("Setting up trainer...")
    
    # 데이터셋 검증
    train_dataset = dataset.get("train", None)
    eval_dataset = dataset.get("test", None)
    
    if train_dataset is None or len(train_dataset) == 0:
        raise ValueError(f"훈련 데이터셋이 비어있습니다! 데이터셋 로딩을 확인하세요.")
    
    print(f"✅ 데이터셋 검증 완료:")
    print(f"  - 훈련 데이터: {len(train_dataset)} 샘플")
    if eval_dataset is not None:
        print(f"  - 평가 데이터: {len(eval_dataset)} 샘플")
    else:
        print(f"  - 평가 데이터: 없음")
    
    # SFTTrainer에서 사용할 수 있도록 데이터셋 형태를 한번 더 확인
    print("데이터셋 샘플 확인:")
    print(f"  - 첫 번째 훈련 샘플 키: {list(train_dataset[0].keys())}")
    print(f"  - 첫 번째 샘플 messages: {train_dataset[0]['messages'][:100]}")
    print(f"  - 첫 번째 샘플 images: {train_dataset[0]['images'][0].size}")

    trainer = SFTTrainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collate_fn,
        optimizers=(custom_optimizer, None) if custom_optimizer is not None else (None, None)
    )
    # Enforce: Disable gradient checkpointing with ZeRO-3 at runtime as an additional safeguard
    try:
        ds_cfg_path = getattr(trainer.args, "deepspeed", None)
        if ds_cfg_path:
            import json
            with open(ds_cfg_path, "r") as f:
                _zero_stage = int((json.load(f).get("zero_optimization", {}) or {}).get("stage", 0) or 0)
            if _zero_stage == 3:
                if hasattr(trainer.model, "gradient_checkpointing_disable"):
                    trainer.model.gradient_checkpointing_disable()
                trainer.args.gradient_checkpointing = False
                print("✓ Disabled gradient checkpointing for DeepSpeed ZeRO-3 compatibility")
    except Exception as _:
        pass
    # Add MoE monitoring callback
    trainer.add_callback(
        create_moe_callback_for_transformers(
            num_experts=model_config.get("g3moe_params", {}).get("n_routed_experts", 8),
            log_every_n_steps=1,             # 매 스텝마다 로그 기록
            logger=wandb,                    # 사용할 로거 지정 (wandb)
            log_to_console=True,             # 콘솔에도 주요 메트릭 출력
            debug_logging=True,              # ✅ 디버그 로깅 활성화
                         # === (선택사항) === #
            log_heatmap_every=5,             # 500 스텝마다 Expert 사용률 히트맵 로깅
            alert_threshold_imbalance=4.0,   # 특정 Expert 사용률이 평균의 4배를 초과하면 경고
            unused_expert_threshold=0.25,    # 25% 이상의 Expert가 미사용되면 경고
            entropy_threshold=0.1,           # 라우팅 엔트로피가 0.1 미만이면 경고
            save_detailed_logs=False  ,      # 상세 JSON 로그 저장 여부
            enable_generation_logging=True,  # 생성 로깅 활성화
        ))
    
    # Add custom training progress callback
    from transformers import TrainerCallback
    class DetailedTrainingCallback(TrainerCallback):
        def __init__(self, logger):
            self.logger = logger
            self.last_log_time = time.time()
            self.log_interval = 10  # Log every 10 seconds during training
            
        def on_step_begin(self, args, state, control, **kwargs):
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                log_training_progress(
                    self.logger, 
                    kwargs.get('trainer'), 
                    step=state.global_step, 
                    epoch=state.epoch)
                log_gpu_memory(self.logger, f"STEP_{state.global_step}")
                self.last_log_time = current_time
                
        def on_step_end(self, args, state, control, **kwargs):
            # Log every 10 steps for detailed monitoring
            if state.global_step % 10 == 0:
                self.logger.debug(f"📊 Step {state.global_step} completed")
                
        def on_epoch_begin(self, args, state, control, **kwargs):
            self.logger.info(f"📅 Starting epoch {int(state.epoch)}")
            log_gpu_memory(self.logger, f"EPOCH_{int(state.epoch)}_START")
            
        def on_epoch_end(self, args, state, control, **kwargs):
            self.logger.info(f"📅 Completed epoch {int(state.epoch)}")
            log_gpu_memory(self.logger, f"EPOCH_{int(state.epoch)}_END")
            
        def on_train_begin(self, args, state, control, **kwargs):
            self.logger.info("🚀 Training started")
            log_gpu_memory(self.logger, "TRAINING_BEGIN")
            
        def on_train_end(self, args, state, control, **kwargs):
            self.logger.info("✅ Training ended")
            log_gpu_memory(self.logger, "TRAINING_END")
            
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                # Log important metrics
                if 'train_loss' in logs:
                    self.logger.info(f"📊 Train Loss: {logs['train_loss']:.6f}")
                if 'learning_rate' in logs:
                    self.logger.debug(f"📊 Learning Rate: {logs['learning_rate']:.2e}")
                if 'grad_norm' in logs:
                    self.logger.debug(f"📊 Gradient Norm: {logs['grad_norm']:.6f}")
    
    trainer.add_callback(DetailedTrainingCallback(logger))
    # trainer.add_callback(
    #     ModelEvalCallback(
    #         trainer=trainer,  # Will be set by Trainer
    #         enable_benchmarks=True,  # Enable benchmark evaluation
    #         benchmarks_to_run=['mmlu', 'hellaswag', 'gsm8k', 'truthfulqa', 'arc', 'piqa'],  # Run multiple benchmarks
    #         benchmark_eval_frequency=training_config["eval_steps"],  # Run benchmarks every 2 epochs
    #         mme_max_samples=10,  # Limit MME samples for faster evaluation
    #     ))
    # trainer.add_callback(
    #     IFEvalCallback(
    #         eval_dataset_name="google/IFEval",
    #         max_samples=100
    #     ))

    # Print training info
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Model: {model_config['model_name_or_path']}")
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
        
        enable_profiler = bool(int(os.getenv("PROFILE_TRAINING", "0")))
        if enable_profiler:
            from torch.profiler import profile, record_function, ProfilerActivity
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                try:
                    trainer.train()
                    profiler_table = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
                    wandb.log({"profiler_table": wandb.Table(data=[profiler_table])})
                except Exception as e:
                    traceback.print_exc()
                    print(f"⚠️ Profiler error: {e}")
        else:
            # eval 최적화를 위한 커스텀 eval 함수 설정
            logger.info("🔧 Setting up memory-optimized evaluation...")
            original_eval_fn = getattr(trainer, 'evaluate', None)
            trainer.evaluate = lambda eval_dataset=None, ignore_keys=None, metric_key_prefix="eval": eval_with_memory_optimization(trainer, original_eval_fn, eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            # Log initial memory state
            log_gpu_memory(logger, "TRAINING_START")
            
            # Start training with progress monitoring
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            logger.info(f"✅ Training completed successfully in {training_time:.2f} seconds")
        
    except KeyboardInterrupt as e:
        logger.error(f"❌ KeyboardInterrupt during training: {str(e)}")
        log_error_context(logger, e, "training_keyboard_interrupt")
        raise e

    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"❌ RuntimeError during training: {error_msg}")
        
        if "CUDA out of memory" in error_msg:
            logger.error("❌ CUDA OOM 발생! 상세 정보를 수집합니다...")
            
            # Log detailed memory state at OOM
            log_gpu_memory(logger, "OOM_ERROR")
            
            # Log training state at OOM
            if hasattr(trainer, 'state') and trainer.state is not None:
                state = trainer.state
                logger.error(f"❌ Training state at OOM:")
                logger.error(f"  - Global step: {state.global_step}")
                logger.error(f"  - Epoch: {state.epoch:.3f}")
                logger.error(f"  - Current loss: {getattr(state, 'log_history', [{}])[-1].get('train_loss', 'N/A')}")
            
            # Log model state
            logger.error(f"❌ Model state at OOM:")
            logger.error(f"  - Model device: {next(trainer.model.parameters()).device}")
            logger.error(f"  - Model dtype: {next(trainer.model.parameters()).dtype}")
            logger.error(f"  - Model requires_grad: {next(trainer.model.parameters()).requires_grad}")
            
            # Log batch information
            if hasattr(trainer, 'train_dataloader'):
                try:
                    batch_size = trainer.per_device_train_batch_size
                    grad_accum = trainer.gradient_accumulation_steps
                    effective_batch = batch_size * grad_accum
                    logger.error(f"❌ Batch configuration at OOM:")
                    logger.error(f"  - Per device batch size: {batch_size}")
                    logger.error(f"  - Gradient accumulation: {grad_accum}")
                    logger.error(f"  - Effective batch size: {effective_batch}")
                except Exception as batch_e:
                    logger.error(f"❌ Could not get batch info: {batch_e}")
            
            logger.error("❌ 메모리 정리 후 재시도...")
            clear_gpu_memory()
            logger.error("❌ GPU 메모리 정리 완료. 다시 시도해주세요.")
            
        else:
            logger.error(f"❌ Other RuntimeError: {error_msg}")
            log_error_context(logger, e, "training_runtime_error")
        
        raise e
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during training: {str(e)}")
        log_error_context(logger, e, "training_unexpected_error")
        raise e
        
    finally:
        # 원래 eval 함수 복원
        # Save final model
        print("Saving final model...")
        if config.get("deepspeed_config") is not None:
            trainer.deepspeed.save_checkpoint(training_args.output_dir)
        trainer.save_model()
        
        # Save tokenizer``
        tokenizer.save_pretrained(training_args.output_dir)
        print("Training End")
        if original_eval_fn:
            logger.debug("🔧 Restoring original evaluation function...")
            trainer.evaluate = original_eval_fn


if __name__ == "__main__":
    register_custom_optimizers()
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="SPECTRA MoE SFT Training with Config File")
        parser.add_argument(
            "--config", 
            type=str, 
            default="config/sft/spectra_training_config.json",
            help="Path to training configuration JSON file"
        )
        args = parser.parse_args()
        
        # Load configuration
        config = load_config(args.config)
        
        model_config = config["model_config"]
        data_config = config["data_config"]
        training_config = config["training_config"]
        
        # Set seed
        set_seed(training_config["seed"])
        # Initialize wandb if needed
        if training_config.get("report_to") and "wandb" in training_config["report_to"]:
            rank = int(os.getenv("RANK", "0"))
            if rank == 0:
                wandb.init(
                    project="SPECTRA-moe-sft",
                    name=training_config["run_name"],
                    config=config
                )

        main(model_config, data_config, training_config)

    except Exception as e:
        logger.error(f"❌ Fatal error in main: {str(e)}")
        log_error_context(logger, e, "main_function")
        
        # Log final memory state
        if torch.cuda.is_available():
            logger.error("❌ Final GPU memory state:")
            logger.error(f"❌ Memory summary:\n{torch.cuda.memory_summary()}")
            logger.error(f"❌ Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            logger.error(f"❌ Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f}GB")
        
        # Re-raise the exception
        raise e
