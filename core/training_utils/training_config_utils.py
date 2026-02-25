"""
Training configuration utilities
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional
from trl import SFTConfig

class CustomSFTConfig(SFTConfig):
    """
    Custom SFTConfig that allows overriding read-only properties like place_model_on_device.
    This is necessary for DeepSpeed TP/ZeRO-2 where we want to skip the Trainer's 
    automatic model.to(device) call.
    """
    _place_model_on_device = None

    @property
    def place_model_on_device(self):
        if self._place_model_on_device is not None:
            return self._place_model_on_device
        # If DeepSpeed is set, always return False to prevent OOM during Trainer initialization
        if self.deepspeed:
            return False
        return True


def create_training_args(
    training_config: Dict[str, Any], 
    deepspeed_config: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> SFTConfig:
    """Create SFTConfig from training configuration"""
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Force save_safetensors=False to handle shared router parameters in MoE
    # This avoids RuntimeError when saving models with global_router shared across layers
    training_config["save_safetensors"] = False
    
    # CRITICAL: Remove deepspeed from training_config if deepspeed_config parameter is provided
    # This prevents SFTConfig from trying to read an invalid deepspeed config during __init__
    training_config_copy = training_config.copy()
    # Always remove deepspeed key to prevent DeepSpeedPlugin from trying to read file during SFTConfig init
    # We'll set it manually after SFTConfig is created
    training_config_copy.pop("deepspeed", None)
    # Also remove any DeepSpeed-related keys that might cause issues
    training_config_copy.pop("deepspeed_config", None)
    training_config_copy.pop("deepspeed_config_file", None)
    
    # Handle place_model_on_device logic
    place_val = training_config_copy.pop("place_model_on_device", None)

    # Create CustomSFTConfig with all parameters (without deepspeed key)
    training_args = CustomSFTConfig(
        **training_config_copy,
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    
    if place_val is not None:
        training_args._place_model_on_device = place_val
    
    # Add DeepSpeed config if provided (AFTER SFTConfig creation to avoid JSON parsing during init)
    if deepspeed_config:
        ds_cfg_path_abs = os.path.abspath(deepspeed_config)
        
        # Validate DeepSpeed config file exists and is valid JSON before setting
        # Use retry logic to handle race conditions and file system delays
        max_retries = 5
        retry_delay = 0.2  # 200ms between retries
        
        for attempt in range(max_retries):
            try:
                # Check if file exists
                if not os.path.exists(ds_cfg_path_abs):
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise FileNotFoundError(f"DeepSpeed config file not found after {max_retries} attempts: {ds_cfg_path_abs}")
                
                # Try to read and validate file
                with open(ds_cfg_path_abs, "r", encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # Check if content is empty
                if not content:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise ValueError(f"DeepSpeed config file is empty after {max_retries} attempts: {ds_cfg_path_abs}")
                
                # Validate JSON syntax with detailed error reporting
                try:
                    json.loads(content)
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
                
                # Success - break out of retry loop
                break
                
            except (FileNotFoundError, ValueError) as e:
                # Re-raise these immediately (no retry)
                raise
            except Exception as e:
                # For other exceptions, retry if attempts remain
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ValueError(f"Error reading DeepSpeed config file {ds_cfg_path_abs} after {max_retries} attempts: {e}")
        
        # Now set deepspeed config (file is validated)
        training_args.deepspeed = ds_cfg_path_abs
        logger.info(f"DeepSpeed config set: {ds_cfg_path_abs}")
        # Validate that CPU offload is disabled as required
        # NOTE: Router learning issues with ZeRO-3 + CPU offload
        # If router weights are not learning, try:
        # 1. Reduce ZeRO stage from 3 to 2 (change "stage": 3 to "stage": 2)
        # 2. Disable CPU offload (set "device": "none" for offload_optimizer and offload_param)
        # 3. These changes help isolate whether the issue is due to parameter partitioning or offloading
        try:
            with open(ds_cfg_path_abs, "r") as f:
                ds_cfg = json.load(f)
            zero = ds_cfg.get("zero_optimization", {})
            off_opt = (zero.get("offload_optimizer") or {}).get("device", "none").lower()
            off_param = (zero.get("offload_param") or {}).get("device", "none").lower()
            zero_stage = zero.get("stage", 0)
            # Param offloading 금지 (param offloading 금지)
            if off_param != "none":
                logger.error("❌ Param offload is forbidden (param offloading 금지). Set zero_optimization.offload_param.device to 'none' in your DeepSpeed config.")
                raise ValueError("Param offload must be disabled (offload_param.device='none'). Param offloading 금지.")
            logger.info(f"DeepSpeed zero stage: {zero_stage}, offload_optimizer: {off_opt}, offload_param: {off_param}")
            if zero_stage == 3 and (off_opt != "none" or off_param != "none"):
                logger.warning("⚠️ WARNING: Using ZeRO-3 with CPU offload may cause router learning issues!")
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
            logger.warning(f"⚠️ DeepSpeed config validation warning: {e}")
    
    return training_args

