"""
Error handling utilities for training
"""
import os
import json
import traceback
import logging
from datetime import datetime
from typing import Optional
import torch


def log_error_context(logger: logging.Logger, error: Exception, context: str = ""):
    """Log detailed error context with system state"""
    logger.error(f"❌ Error in {context}: {str(error)}")
    logger.error(f"❌ Error type: {type(error).__name__}")
    
    # Log traceback
    logger.error(f"❌ Traceback:\n{traceback.format_exc()}")
    
    # Log GPU memory state
    if torch.cuda.is_available():
        from core.training_utils.logging_utils import log_gpu_memory
        memory_info = log_gpu_memory(logger, "ERROR")
        if memory_info:
            logger.error(f"❌ GPU Memory at error - Allocated: {memory_info['allocated']:.2f}GB, Reserved: {memory_info['reserved']:.2f}GB")
    
    # Log system state
    logger.error(f"❌ System state - CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logger.error(f"❌ Current device: {torch.cuda.current_device()}, Device name: {torch.cuda.get_device_name()}")


def save_oom_error_info(
    logger: logging.Logger, 
    trainer, 
    error: Exception, 
    batch_info: Optional[dict] = None, 
    output_dir: Optional[str] = None
) -> Optional[str]:
    """OOM 에러 정보 저장"""
    try:
        if output_dir is None:
            output_dir = getattr(trainer.args, 'output_dir', None) if trainer and hasattr(trainer, 'args') else "logs"
        os.makedirs(output_dir, exist_ok=True)
        error_file = os.path.join(output_dir, f"oom_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        from core.training_utils.logging_utils import log_gpu_memory
        oom_info = {
            'timestamp': datetime.now().isoformat(),
            'error': {'type': type(error).__name__, 'message': str(error)},
            'gpu_memory': log_gpu_memory(logger, "OOM") if torch.cuda.is_available() else {}
        }
        
        if batch_info:
            oom_info['batch_info'] = batch_info
        
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(oom_info, f, indent=2, ensure_ascii=False, default=str)
        logger.error(f"💾 OOM 에러 정보 저장: {error_file}")
        return error_file
    except Exception as e:
        logger.error(f"❌ OOM 에러 정보 저장 실패: {e}")
        return None


def handle_cuda_oom(e: torch.OutOfMemoryError, trainer, logger: logging.Logger):
    """CUDA OOM 처리"""
    logger.error(f"❌ CUDA OOM: {str(e)}")
    from core.training_utils.logging_utils import log_gpu_memory
    log_gpu_memory(logger, "OOM")
    if hasattr(trainer, 'state') and trainer.state:
        epoch_str = f"{trainer.state.epoch:.3f}" if trainer.state.epoch is not None else "N/A"
        logger.error(f"Step: {trainer.state.global_step}, Epoch: {epoch_str}")
    save_oom_error_info(logger, trainer, e)
    from core.training_utils.memory_utils import clear_gpu_memory
    clear_gpu_memory(logger)
    logger.error("💡 해결 방법: batch_size 감소, gradient_accumulation_steps 증가, max_length 감소")


def handle_ram_oom(e: MemoryError, trainer, logger: logging.Logger):
    """RAM OOM 처리"""
    logger.error(f"❌ RAM OOM: {str(e)}")
    save_oom_error_info(logger, trainer, e)
    logger.error("💡 해결 방법: streaming=True, 배치 크기 감소, CPU offload 활성화")


def handle_training_exception(
    e: Exception, 
    trainer, 
    logger: logging.Logger, 
    context: str = "training"
):
    """
    학습 중 발생하는 일반 exception을 통합 처리하는 함수
    
    Args:
        e: Exception 객체
        trainer: Trainer 객체
        logger: Logger 객체
        context: 에러 발생 컨텍스트 (예: "training", "training_keyboard_interrupt", "training_runtime_error")
    """
    error_msg = str(e)
    error_type = type(e).__name__
    
    logger.error(f"❌ {error_type} during {context}: {error_msg}")
    log_error_context(logger, e, context)
    
    # 특정 에러 타입별 추가 처리
    if isinstance(e, KeyboardInterrupt):
        logger.error("❌ 학습이 사용자에 의해 중단되었습니다.")
    elif isinstance(e, RuntimeError):
        # CUBLAS 메모리 할당 실패 등 RuntimeError 처리
        if "CUBLAS_STATUS_ALLOC_FAILED" in error_msg or "cublasCreate" in error_msg:
            logger.error("❌ CUBLAS 메모리 할당 실패 - GPU 메모리 문제일 수 있습니다.")
            from core.training_utils.logging_utils import log_gpu_memory
            log_gpu_memory(logger, "CUBLAS_ERROR")
        # NCCL 오류 처리
        elif "NCCL" in error_msg or "nccl" in error_msg.lower() or "DistBackendError" in error_type:
            logger.error("❌ NCCL 분산 통신 오류가 발생했습니다.")
            logger.error("   가능한 원인:")
            logger.error("   1. 네트워크 연결 문제")
            logger.error("   2. 원격 프로세스가 예기치 않게 종료됨")
            logger.error("   3. GPU 간 통신 문제")
            logger.error("   4. DeepSpeed 초기화 실패")
            logger.error("   💡 해결 방법:")
            logger.error("   - 네트워크 상태 확인")
            logger.error("   - 모든 노드가 정상 작동하는지 확인")
            logger.error("   - NCCL 환경 변수 조정 (NCCL_DEBUG=INFO)")
            logger.error("   - 단일 GPU로 테스트")
        else:
            logger.error(f"❌ RuntimeError: {error_msg}")
    else:
        logger.error(f"❌ Unexpected {error_type}: {error_msg}")


def collect_environment_info() -> dict:
    """간단한 환경 정보 수집"""
    env_info = {'timestamp': datetime.now().isoformat()}
    try:
        env_info['pytorch'] = {'version': torch.__version__, 'cuda_available': torch.cuda.is_available()}
        if torch.cuda.is_available():
            env_info['cuda'] = {
                'device_count': torch.cuda.device_count(),
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3
            }
    except Exception as e:
        env_info['error'] = str(e)
    return env_info

