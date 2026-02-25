"""
Logging utilities for training
"""
import os
import sys
import logging
from datetime import datetime
from typing import Optional


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging system for training monitoring
    
    Only main process (rank 0) outputs to console.
    All processes write to log files.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Check if this is the main process (rank 0)
    # Works with DeepSpeed, Accelerate, and torchrun
    is_main_process = False
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            is_main_process = (dist.get_rank() == 0)
        else:
            # Fallback to environment variables if dist not initialized yet
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
            is_main_process = (rank == 0)
    except:
        # Fallback to environment variables
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        is_main_process = (rank == 0)
    
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    
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
    
    # File handler for detailed logs (all ranks write to their own files)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"training_detailed_{timestamp}_rank{local_rank}.log"),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # File handler for error logs
    error_handler = logging.FileHandler(
        os.path.join(log_dir, f"training_errors_{timestamp}_rank{local_rank}.log"),
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # Console handler for important messages - ONLY FOR MAIN PROCESS
    if is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    else:
        # Non-main processes: suppress all console output
        # Still write to files for debugging if needed
        pass
    
    return logger


def log_gpu_memory(logger: logging.Logger, stage: str, device: int = 0) -> Optional[dict]:
    """Log detailed GPU memory information"""
    import torch
    
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


def log_training_progress(
    logger: logging.Logger, 
    trainer, 
    step: Optional[int] = None, 
    epoch: Optional[float] = None, 
    loss: Optional[float] = None
):
    """Log training progress"""
    if hasattr(trainer, 'state') and trainer.state is not None:
        state = trainer.state
        current_step = step or state.global_step
        current_epoch = epoch or state.epoch
        current_loss = loss or getattr(state, 'log_history', [{}])[-1].get('train_loss', 'N/A')
        logger.info(f"Step: {current_step}, Epoch: {current_epoch:.3f}, Loss: {current_loss}")

