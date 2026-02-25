"""
Router parameter utilities for training
"""
import logging
from typing import Tuple, List
from models.spectra_model import SPECTRARouter


def ensure_router_parameters_trainable(
    model, 
    logger: logging.Logger, 
    context: str = ""
) -> Tuple[List, List, int]:
    """
    Router 파라미터를 trainable로 설정하는 통합 함수 (중복 코드 제거)
    
    Args:
        model: 모델 인스턴스 (DeepSpeed 래핑 가능)
        logger: 로거 인스턴스
        context: 컨텍스트 문자열 (로그용)
    
    Returns:
        tuple: (router_params_list, router_param_names_list, trainable_count)
    """
    # 실제 모델 추출 (DeepSpeed 래핑 처리)
    actual_model = model
    if hasattr(model, 'module'):  # DeepSpeed 래핑
        actual_model = model.module
    
    router_params = []
    router_param_names = []
    seen_param_ids = set()
    
    for name, module in actual_model.named_modules():
        if isinstance(module, (SPECTRARouter)):
            if context:
                logger.debug(f"  [{context}] Found router module: {name}")
            
            # Router 모듈의 모든 파라미터
            for param_name, param in module.named_parameters(recurse=True):
                full_name = f"{name}.{param_name}"
                param_id = id(param)
                if param_id not in seen_param_ids:
                    router_params.append(param)
                    router_param_names.append(full_name)
                    seen_param_ids.add(param_id)
                if not param.requires_grad:
                    param.requires_grad_(True)
                    if context:
                        logger.debug(f"    [{context}] Set requires_grad=True: {full_name}")
            
            # Expression projector 파라미터
            if hasattr(module, 'expression_projector'):
                expr_proj = module.expression_projector
                for ep_param_name, ep_param in expr_proj.named_parameters(recurse=True):
                    full_name = f"{name}.expression_projector.{ep_param_name}"
                    ep_param_id = id(ep_param)
                    if ep_param_id not in seen_param_ids:
                        router_params.append(ep_param)
                        router_param_names.append(full_name)
                        seen_param_ids.add(ep_param_id)
                    if not ep_param.requires_grad:
                        ep_param.requires_grad_(True)
                        if context:
                            logger.debug(f"      [{context}] Set requires_grad=True: {full_name}")
            
            # Load balancer 파라미터
            if hasattr(module, 'load_balancer'):
                lb_module = module.load_balancer
                for lb_param_name, lb_param in lb_module.named_parameters(recurse=True):
                    full_name = f"{name}.load_balancer.{lb_param_name}"
                    lb_param_id = id(lb_param)
                    if lb_param_id not in seen_param_ids:
                        router_params.append(lb_param)
                        router_param_names.append(full_name)
                        seen_param_ids.add(lb_param_id)
                    if not lb_param.requires_grad:
                        lb_param.requires_grad_(True)
                        if context:
                            logger.debug(f"      [{context}] Set requires_grad=True: {full_name}")
    
    trainable_count = sum(1 for p in router_params if p.requires_grad)
    
    if context:
        logger.info(f"  [{context}] Router parameters: {trainable_count}/{len(router_params)} trainable")
    
    return router_params, router_param_names, trainable_count


def ensure_router_in_optimizer(trainer, model, logger: logging.Logger, modules_to_save_list=None):
    """Router 파라미터가 올바르게 학습 가능한지 검증하고 필요시 수정"""
    try:
        # 통합 함수 사용 (중복 코드 제거)
        router_params, router_param_names, trainable_count = ensure_router_parameters_trainable(
            model, logger, context="optimizer_validation"
        )
        
        if not router_params:
            logger.error("❌ CRITICAL: No router parameters found in model!")
            return
        
        logger.debug(f"✅ Found {len(router_params)} router parameters")
        logger.debug(f"✅ Router parameters trainable: {trainable_count}/{len(router_params)}")
        
        # Optimizer 포함 여부 확인 및 추가
        if hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
            logger.info("✅ DeepSpeed detected - router params with requires_grad=True will be included automatically")
            if hasattr(trainer.deepspeed, 'optimizer') and trainer.deepspeed.optimizer is not None:
                ds_optimizer = trainer.deepspeed.optimizer
                if hasattr(ds_optimizer, 'param_groups'):
                    ds_param_ids = {id(p) for group in ds_optimizer.param_groups for p in group['params']}
                    router_param_ids = {id(p) for p in router_params}
                    in_ds_optimizer = router_param_ids & ds_param_ids
                    logger.debug(f"   Router params in DeepSpeed optimizer: {len(in_ds_optimizer)}/{len(router_params)}")
        
        elif hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            optimizer_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
            router_param_ids = {id(p) for p in router_params}
            in_optimizer = router_param_ids & optimizer_param_ids
            logger.debug(f"✅ Router params in optimizer: {len(in_optimizer)}/{len(router_params)}")
            
            if len(in_optimizer) < len(router_params):
                missing_params = [p for p in router_params if id(p) not in optimizer_param_ids]
                if len(trainer.optimizer.param_groups) > 0:
                    trainer.optimizer.param_groups[0]['params'].extend(missing_params)
                    logger.debug(f"  ✓ Added {len(missing_params)} parameters to optimizer")
        else:
            logger.warning("⚠️ Optimizer not yet initialized - will be checked after training starts")
    
    except Exception as e:
        import traceback
        logger.error(f"❌ Error validating router weights: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")

