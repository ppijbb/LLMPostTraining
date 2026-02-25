"""
Custom callbacks for training
"""
import os
import json
import time
import types
import logging
import traceback
from typing import Optional
from transformers import TrainerCallback
import torch


class BatchTrackingCallback(TrainerCallback):
    """배치 정보를 추적하여 OOM 발생 시 디버깅 정보 제공"""
    def __init__(self, trainer_ref):
        self.last_batch_info = None
        self.last_batch_step = -1
        self.trainer_ref = trainer_ref  # Trainer 참조
    
    def on_train_batch_begin(self, args, state, control, model=None, inputs=None, **kwargs):
        """배치 시작 시 배치 정보 저장 - 가장 확실한 방법"""
        try:
            if inputs is not None:
                trainer = kwargs.get('trainer') or self.trainer_ref
                self._save_batch_info(inputs, state.global_step, trainer)
        except Exception:
            pass  # 배치 정보 저장 실패해도 학습은 계속
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Step 시작 시 배치 정보 저장 시도 (fallback)"""
        try:
            # Trainer의 내부 상태에서 배치 확인
            trainer = kwargs.get('trainer') or self.trainer_ref
            if trainer is not None:
                # Trainer의 _current_batch 또는 최근 배치 확인
                if hasattr(trainer, '_current_batch') and trainer._current_batch is not None:
                    self._save_batch_info(trainer._current_batch, state.global_step, trainer)
        except Exception:
            pass  # 배치 정보 저장 실패해도 학습은 계속
    
    def on_step_end(self, args, state, control, **kwargs):
        """Step 종료 시 배치 정보 저장 시도 (fallback)"""
        try:
            trainer = kwargs.get('trainer') or self.trainer_ref
            if trainer is not None:
                # Trainer의 내부 상태에서 배치 확인
                if hasattr(trainer, '_current_batch') and trainer._current_batch is not None:
                    self._save_batch_info(trainer._current_batch, state.global_step, trainer)
        except Exception:
            pass
    
    def _save_batch_info(self, batch, step, trainer):
        """배치 정보를 메모리 효율적으로 저장"""
        try:
            batch_info = {}
            
            # Trainer 설정에서 배치 정보 가져오기
            if trainer is not None:
                try:
                    batch_info['per_device_batch_size'] = getattr(trainer, 'per_device_train_batch_size', None)
                    batch_info['gradient_accumulation_steps'] = getattr(trainer, 'gradient_accumulation_steps', None)
                    batch_info['num_devices'] = getattr(trainer.args, 'world_size', 1) if hasattr(trainer, 'args') else 1
                    if batch_info['per_device_batch_size'] and batch_info['gradient_accumulation_steps']:
                        batch_info['effective_batch_size'] = batch_info['per_device_batch_size'] * batch_info['gradient_accumulation_steps'] * batch_info['num_devices']
                    
                    # DataLoader에서 실제 배치 크기 확인
                    try:
                        train_dataloader = trainer.get_train_dataloader()
                        if hasattr(train_dataloader, 'batch_size'):
                            batch_info['dataloader_batch_size'] = train_dataloader.batch_size
                        elif hasattr(train_dataloader, 'batch_sampler') and hasattr(train_dataloader.batch_sampler, 'batch_size'):
                            batch_info['dataloader_batch_size'] = train_dataloader.batch_sampler.batch_size
                    except Exception:
                        pass
                except Exception:
                    pass
            
            # Input IDs 정보
            if 'input_ids' in batch and torch.is_tensor(batch['input_ids']):
                input_ids = batch['input_ids']
                batch_info['input_ids_shape'] = list(input_ids.shape)
                if len(input_ids.shape) > 1:
                    # 각 샘플의 실제 길이 (pad 제외)
                    pad_token_id = 0
                    # processing_class에서 tokenizer 가져오기 (deprecated된 tokenizer 대신)
                    processing_class = getattr(trainer, 'processing_class', None)
                    if processing_class is not None:
                        # AutoProcessor인 경우 tokenizer 속성에 접근
                        tokenizer = getattr(processing_class, 'tokenizer', processing_class)
                        pad_token_id = getattr(tokenizer, 'pad_token_id', 0) or getattr(tokenizer, 'eos_token_id', 0)
                    
                    sample_lengths = (input_ids != pad_token_id).sum(dim=1).cpu().tolist()
                    batch_info['sample_lengths'] = sample_lengths[:10]  # 최대 10개만
                    batch_info['max_length'] = max(sample_lengths) if sample_lengths else 0
                    batch_info['min_length'] = min(sample_lengths) if sample_lengths else 0
                    batch_info['avg_length'] = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0
                    batch_info['total_tokens'] = input_ids.numel()
            
            # Attention mask 정보
            if 'attention_mask' in batch and torch.is_tensor(batch['attention_mask']):
                attn_mask = batch['attention_mask']
                batch_info['attention_mask_shape'] = list(attn_mask.shape)
                batch_info['attention_mask_total'] = attn_mask.numel()
            
            # Pixel values (이미지) 정보
            if 'pixel_values' in batch and torch.is_tensor(batch['pixel_values']):
                pixel_values = batch['pixel_values']
                batch_info['pixel_values_shape'] = list(pixel_values.shape)
                batch_info['pixel_values_dtype'] = str(pixel_values.dtype)
                batch_info['pixel_values_memory_mb'] = pixel_values.numel() * pixel_values.element_size() / 1024 / 1024
                batch_info['num_images'] = pixel_values.shape[0] if len(pixel_values.shape) > 0 else 0
            
            # Image grid 정보
            if 'image_grid_thw' in batch:
                batch_info['image_grid_thw'] = batch['image_grid_thw']
            
            # Labels 정보
            if 'labels' in batch and torch.is_tensor(batch['labels']):
                labels = batch['labels']
                batch_info['labels_shape'] = list(labels.shape)
                if labels.numel() > 0:
                    non_ignore = (labels != -100).sum().item()
                    batch_info['non_ignore_tokens'] = non_ignore
                    batch_info['ignore_tokens'] = (labels == -100).sum().item()
            
            # 실제 배치 크기 (텐서에서 직접 확인)
            if 'input_ids' in batch and torch.is_tensor(batch['input_ids']):
                batch_info['actual_batch_size'] = batch['input_ids'].shape[0] if len(batch['input_ids'].shape) > 0 else 1
                # 기존 batch_size 필드도 유지 (하위 호환성)
                batch_info['batch_size'] = batch_info['actual_batch_size']
            
            self.last_batch_info = batch_info
            self.last_batch_step = step
        except Exception as e:
            pass  # 배치 정보 저장 실패해도 학습은 계속


class DetailedTrainingCallback(TrainerCallback):
    """상세한 트레이닝 진행 상황 로깅"""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_log_time = time.time()
        self.log_interval = 10  # Log every 10 seconds during training
    
    def on_step_begin(self, args, state, control, **kwargs):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            from core.training_utils.logging_utils import log_training_progress, log_gpu_memory
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
        from core.training_utils.logging_utils import log_gpu_memory
        log_gpu_memory(self.logger, f"EPOCH_{int(state.epoch)}_START")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.logger.info(f"📅 Completed epoch {int(state.epoch)}")
        from core.training_utils.logging_utils import log_gpu_memory
        log_gpu_memory(self.logger, f"EPOCH_{int(state.epoch)}_END")
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.logger.info("🚀 Training started")
        from core.training_utils.logging_utils import log_gpu_memory
        log_gpu_memory(self.logger, "TRAINING_BEGIN")
    
    def on_train_end(self, args, state, control, **kwargs):
        self.logger.info("✅ Training ended")
        from core.training_utils.logging_utils import log_gpu_memory
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


class ModulesToSaveSyncCallback(TrainerCallback):
    """original_module.*의 값을 modules_to_save.default.*로 동기화"""
    def __init__(self, sync_every_n_steps: int = 10, logger: Optional[logging.Logger] = None):
        self.sync_every_n_steps = sync_every_n_steps
        self.last_sync_step = -1
        self.logger = logger or logging.getLogger(__name__)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """각 step 끝에서 동기화 (주기적으로)"""
        if state.global_step % self.sync_every_n_steps == 0 and state.global_step > self.last_sync_step:
            try:
                actual_model = model
                if hasattr(model, 'module'):  # DeepSpeed 래핑
                    actual_model = model.module
                
                if actual_model is None:
                    return control
                
                from models.spectra_model import SPECTRARouter, SwitchRouterAdapter
                sync_count = 0
                
                for name, module in actual_model.named_modules():
                    if isinstance(module, SPECTRARouter):
                        # expression_projector의 linear_projection 동기화
                        if hasattr(module, 'expression_projector'):
                            expr_proj = module.expression_projector
                            if hasattr(expr_proj, 'linear_projection'):
                                lin_proj = expr_proj.linear_projection
                                
                                # PEFT ModulesToSaveWrapper 확인
                                if hasattr(lin_proj, 'original_module') and hasattr(lin_proj, 'modules_to_save'):
                                    if hasattr(lin_proj.modules_to_save, 'default'):
                                        default_module = lin_proj.modules_to_save.default
                                        
                                        # original_module의 파라미터를 modules_to_save.default로 복사
                                        for orig_param_name, orig_param in lin_proj.original_module.named_parameters(recurse=True):
                                            if hasattr(default_module, orig_param_name):
                                                default_param = getattr(default_module, orig_param_name)
                                                if orig_param.shape == default_param.shape:
                                                    with torch.no_grad():
                                                        default_param.data.copy_(orig_param.data)
                                                    sync_count += 1
                                
                                # 또는 modules_to_save.default가 직접 있는 경우
                                elif hasattr(lin_proj, 'modules_to_save') and hasattr(lin_proj.modules_to_save, 'default'):
                                    # 이 경우는 이미 modules_to_save.default가 forward에서 사용되는 경우
                                    pass
                
                if sync_count > 0:
                    self.logger.debug(f"✅ Synced {sync_count} router parameters from original_module to modules_to_save.default at step {state.global_step}")
                
                self.last_sync_step = state.global_step
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to sync modules_to_save at step {state.global_step}: {e}")
        
        return control
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Checkpoint 저장 전에 동기화 (저장 직전)"""
        try:
            actual_model = model
            if hasattr(model, 'module'):  # DeepSpeed 래핑
                actual_model = model.module
            
            if actual_model is None:
                return control
            
            from models.spectra_model import SPECTRARouter, SwitchRouterAdapter
            sync_count = 0
            
            for name, module in actual_model.named_modules():
                if isinstance(module, SPECTRARouter):
                    # expression_projector의 linear_projection 동기화
                    if hasattr(module, 'expression_projector'):
                        expr_proj = module.expression_projector
                        if hasattr(expr_proj, 'linear_projection'):
                            lin_proj = expr_proj.linear_projection
                            
                            # PEFT ModulesToSaveWrapper 확인
                            if hasattr(lin_proj, 'original_module') and hasattr(lin_proj, 'modules_to_save'):
                                if hasattr(lin_proj.modules_to_save, 'default'):
                                    default_module = lin_proj.modules_to_save.default
                                    
                                    # original_module의 파라미터를 modules_to_save.default로 복사
                                    for orig_param_name, orig_param in lin_proj.original_module.named_parameters(recurse=True):
                                        if hasattr(default_module, orig_param_name):
                                            default_param = getattr(default_module, orig_param_name)
                                            if orig_param.shape == default_param.shape:
                                                with torch.no_grad():
                                                    default_param.data.copy_(orig_param.data)
                                                sync_count += 1
            
            if sync_count > 0:
                self.logger.info(f"✅ Synced {sync_count} router parameters from original_module to modules_to_save.default before save at step {state.global_step}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to sync modules_to_save before save at step {state.global_step}: {e}")
        
        return control


class FrozenParamGradInitCallback(TrainerCallback):
    """Frozen 파라미터의 gradient를 초기화하는 콜백"""
    def __init__(self, trainer_ref, logger: Optional[logging.Logger] = None):
        self.initialized = False
        self.trainer_ref = trainer_ref
        self.patch_applied = False
        self.logger = logger or logging.getLogger(__name__)
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # CRITICAL: Apply DeepSpeed gradient patch when training begins
        # DeepSpeed is initialized at this point
        if not self.patch_applied:
            trainer = self.trainer_ref
            if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
                self.logger.info("🔧 DeepSpeed detected in on_train_begin - applying gradient patch for frozen parameters...")
                try:
                    actual_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                    frozen_param_count = sum(1 for p in actual_model.named_parameters() if not p[1].requires_grad)
                    
                    if frozen_param_count > 0:
                        self.logger.info(f"✅ Found {frozen_param_count} frozen parameters - applying DeepSpeed patches...")
                        optimizer = trainer.deepspeed.optimizer
                        patched_methods = []
                        
                        # Patch reduce_independent_p_g_buckets_and_remove_grads
                        if hasattr(optimizer, 'reduce_independent_p_g_buckets_and_remove_grads'):
                            original_reduce_independent = optimizer.reduce_independent_p_g_buckets_and_remove_grads
                            def patched_reduce_independent(self, param, i):
                                if param.grad is None:
                                    param.grad = torch.zeros_like(param)
                                    param.grad.requires_grad_(False)
                                return original_reduce_independent(self, param, i)
                            optimizer.reduce_independent_p_g_buckets_and_remove_grads = types.MethodType(patched_reduce_independent, optimizer)
                            patched_methods.append('reduce_independent_p_g_buckets_and_remove_grads')
                        
                        # Patch reduce_ready_partitions_and_remove_grads
                        if hasattr(optimizer, 'reduce_ready_partitions_and_remove_grads'):
                            original_reduce_ready = optimizer.reduce_ready_partitions_and_remove_grads
                            def patched_reduce_ready(self, param, i):
                                if param.grad is None:
                                    param.grad = torch.zeros_like(param)
                                    param.grad.requires_grad_(False)
                                return original_reduce_ready(self, param, i)
                            optimizer.reduce_ready_partitions_and_remove_grads = types.MethodType(patched_reduce_ready, optimizer)
                            patched_methods.append('reduce_ready_partitions_and_remove_grads')
                        
                        # Patch reduce_partition_and_remove_grads
                        if hasattr(optimizer, 'reduce_partition_and_remove_grads'):
                            original_reduce_partition = optimizer.reduce_partition_and_remove_grads
                            def patched_reduce_partition(self, param):
                                if param.grad is None:
                                    param.grad = torch.zeros_like(param)
                                    param.grad.requires_grad_(False)
                                return original_reduce_partition(param)
                            optimizer.reduce_partition_and_remove_grads = types.MethodType(patched_reduce_partition, optimizer)
                            patched_methods.append('reduce_partition_and_remove_grads')
                        
                        if patched_methods:
                            self.logger.info(f"✅ Patched DeepSpeed methods: {', '.join(patched_methods)}")
                            self.patch_applied = True
                        else:
                            self.logger.warning("⚠️ Could not patch any DeepSpeed reduce methods")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to apply DeepSpeed gradient patch in on_train_begin: {e}")
                    self.logger.debug(traceback.format_exc())
    
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        # CRITICAL: Initialize gradients for all frozen parameters at step begin
        # This runs BEFORE backward, ensuring gradients are ready when DeepSpeed accesses them
        actual_model = model.module if hasattr(model, 'module') else model
        if actual_model is None:
            return
        
        init_count = 0
        for name, param in actual_model.named_parameters():
            if not param.requires_grad:
                # Always ensure grad is not None for frozen parameters
                # This must happen before backward pass
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                    param.grad.requires_grad_(False)
                    init_count += 1
        
        if init_count > 0 and state.global_step <= 3:
            self.logger.debug(f"   Initialized {init_count} frozen parameter gradients at step {state.global_step} (on_step_begin)")
    
    def on_backward(self, args, state, control, model=None, **kwargs):
        # CRITICAL: Re-initialize gradients immediately after backward
        # This runs right after loss.backward() but before DeepSpeed processes gradients
        # DeepSpeed may access param.grad during backward, so we need to ensure it's never None
        actual_model = model.module if hasattr(model, 'module') else model
        if actual_model is None:
            return
        
        init_count = 0
        for name, param in actual_model.named_parameters():
            if not param.requires_grad:
                # Ensure grad is never None, even after backward
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                    param.grad.requires_grad_(False)
                    init_count += 1
        
        if init_count > 0 and state.global_step <= 3:
            self.logger.debug(f"   Re-initialized {init_count} frozen parameter gradients at step {state.global_step} (on_backward)")
        
        # This is a safety net - the real fix is the patch and pre-backward initialization
        for name, param in actual_model.named_parameters():
            if not param.requires_grad and param.grad is None:
                param.grad = torch.zeros_like(param)
                param.grad.requires_grad_(False)

