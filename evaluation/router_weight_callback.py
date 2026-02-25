# coding=utf-8
"""
Router Weight Tracking Callback for Transformers Trainer

이 callback은 transformers Trainer와 통합되어 step별로 router 가중치를 tracking합니다.
"""

import os
import logging
from typing import Optional, Dict, Any
import torch
import numpy as np
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

try:
    from peft.utils.other import ModulesToSaveWrapper
except ImportError:
    ModulesToSaveWrapper = None

from eval.router_weight_tracker import RouterWeightTracker

logger = logging.getLogger(__name__)


class RouterWeightTrackingCallback(TrainerCallback):
    """
    Transformers Trainer용 Router 가중치 tracking callback
    
    사용 예시:
        from eval.router_weight_callback import RouterWeightTrackingCallback
        
        callback = RouterWeightTrackingCallback(
            save_dir="./router_weight_logs",
            log_every_n_steps=100,
        )
        trainer.add_callback(callback)
    """
    
    def __init__(
        self,
        save_dir: str = "./router_weight_logs",
        log_every_n_steps: int = 1,
        save_full_weights: bool = False,
        max_history: int = 1000,
        verbose: bool = True,
        check_weight_change: bool = True,
        min_change_threshold: float = 1e-8,
        check_after_steps: int = 10,
    ):
        """
        Args:
            save_dir: 가중치 로그 저장 디렉토리
            log_every_n_steps: N step마다 로그 저장
            save_full_weights: True면 전체 가중치 텐서 저장 (메모리 많이 사용)
            max_history: 메모리에 유지할 최대 step 수
            verbose: 상세 로깅 여부
            check_weight_change: weight 변화 체크 여부
            min_change_threshold: 최소 변화 임계값 (이보다 작으면 변화 없음으로 간주)
            check_after_steps: 몇 step 후부터 변화 체크 시작
        """
        self.save_dir = save_dir
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose
        self.check_weight_change = check_weight_change
        self.min_change_threshold = min_change_threshold
        self.check_after_steps = check_after_steps
        
        # 디렉토리 생성 확인
        os.makedirs(save_dir, exist_ok=True)
        if self.verbose:
            logger.info(f"✅ RouterWeightTrackingCallback initialized: save_dir={save_dir}, log_every_n_steps={log_every_n_steps}")
            if self.check_weight_change:
                logger.info(f"   Weight change checking enabled: threshold={min_change_threshold}, check_after_steps={check_after_steps}")
        
        self.tracker = RouterWeightTracker(
            save_dir=save_dir,
            save_every_n_steps=log_every_n_steps,
            save_full_weights=save_full_weights,
            max_history=max_history,
        )
        
        self._first_step_logged = False
        self._last_weight_changes = {}  # layer별 마지막 변화량 저장
        self._last_trainer = None  # 디버깅용 trainer 참조
        self._optimizer_validation_done = False  # Optimizer 검증 완료 플래그 (DeepSpeed lazy init 대응)
        self._router_forward_tracker = {}  # step별로 사용된 router 추적: {step: [router_names]}
        self._router_hooks = []  # 등록된 forward hook들 (나중에 제거용)
        self._actual_router_weights = {}  # Forward hook에서 추적한 실제 사용되는 router weight: {router_name: {step: weight_tensor}}
        self._prev_actual_weights = {}  # 이전 step의 실제 사용된 weight: {router_name: weight_tensor} (직접 비교용)
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Training 시작 시 router 파라미터 검증 (requires_grad는 변경하지 않음)"""
        logger.info("=" * 80)
        logger.info("🔧 Router weight tracking callback initialized (requires_grad 변경하지 않음)")
        logger.info("=" * 80)
        
        # ✅ Trainer 참조 저장 (디버깅 및 검증용)
        trainer = kwargs.get('trainer')
        if trainer is not None:
            self._last_trainer = trainer
        
        # 모델이 없으면 에러
        if model is None:
            logger.error("❌ Model is None in on_train_begin - cannot set router parameters")
            return control
        
        # Router 파라미터 강제로 requires_grad=True 설정 (trainer 없어도 모델에서 직접 설정)
        actual_model = model
        if hasattr(model, 'module'):  # DeepSpeed 래핑
            actual_model = model.module
        
        if actual_model is not None:
            from models.spectra_model import SPECTRARouter
            try:
                from models.g3moe_model import G3MoERouter, G3MoEGRINMoE
            except ImportError:
                G3MoERouter = None
                G3MoEGRINMoE = None
            
            fixed_count = 0
            router_modules_found = []
            seen_router_ids = set()  # 이미 처리한 router 인스턴스 추적
            
            for name, module in actual_model.named_modules():
                is_router = False
                router_module = None
                
                # 1. PEFT ModulesToSaveWrapper 체크 (가장 중요)
                if ModulesToSaveWrapper is not None and isinstance(module, ModulesToSaveWrapper):
                    active_adapter = getattr(module, "active_adapter", "default")
                    if hasattr(module, "modules_to_save") and active_adapter in module.modules_to_save:
                        inner_module = module.modules_to_save[active_adapter]
                        if isinstance(inner_module, SPECTRARouter):
                            is_router = True
                            router_module = inner_module
                        elif G3MoERouter is not None and isinstance(inner_module, G3MoERouter):
                            is_router = True
                            router_module = inner_module
                
                # 2. SPECTRARouter 체크
                elif isinstance(module, SPECTRARouter):
                    is_router = True
                    router_module = module
                # 3. G3MoERouter 직접 체크
                elif G3MoERouter is not None and isinstance(module, G3MoERouter):
                    is_router = True
                    router_module = module
                # 4. G3MoEGRINMoE 내부의 router 속성 체크
                elif G3MoEGRINMoE is not None and isinstance(module, G3MoEGRINMoE):
                    if hasattr(module, 'router'):
                        potential_router = module.router
                        
                        if ModulesToSaveWrapper is not None and isinstance(potential_router, ModulesToSaveWrapper):
                            active_adapter = getattr(potential_router, "active_adapter", "default")
                            if hasattr(potential_router, "modules_to_save") and active_adapter in potential_router.modules_to_save:
                                inner_module = potential_router.modules_to_save[active_adapter]
                                if isinstance(inner_module, G3MoERouter):
                                    is_router = True
                                    router_module = inner_module
                                    name = f"{name}.router"
                        
                        elif isinstance(potential_router, G3MoERouter):
                            is_router = True
                            router_module = potential_router
                            name = f"{name}.router"
                            
                # 5. 일반적인 router 구조 체크 (load_balancer + expression_projector)
                elif hasattr(module, 'load_balancer') and hasattr(module, 'expression_projector'):
                    is_router = True
                    router_module = module
                
                if is_router and router_module is not None:
                    # 같은 router 인스턴스는 한 번만 처리
                    router_id = id(router_module)
                    if router_id in seen_router_ids:
                        continue
                    seen_router_ids.add(router_id)
                    
                    router_modules_found.append(name)
                    # Load balancer 파라미터 - requires_grad 변경하지 않음 (학습에 영향 주지 않도록)
                    # if hasattr(router_module, 'load_balancer'):
                    #     for param_name, param in router_module.load_balancer.named_parameters(recurse=True):
                    #         if not param.requires_grad:
                    #             param.requires_grad_(True)
                    #             fixed_count += 1
                    #             logger.info(f"  ✓ Set requires_grad=True: {name}.load_balancer.{param_name}")
                    
                    # Expression projector 파라미터 - requires_grad 변경하지 않음 (학습에 영향 주지 않도록)
                    # if hasattr(router_module, 'expression_projector'):
                    #     expr_proj = router_module.expression_projector
                    #     for param_name, param in expr_proj.named_parameters(recurse=True):
                    #         if not param.requires_grad:
                    #             param.requires_grad_(True)
                    #             fixed_count += 1
                    #             logger.info(f"  ✓ Set requires_grad=True: {name}.expression_projector.{param_name}")
                    #     
                    #     # linear_projection이 별도로 있는 경우
                    #     if hasattr(expr_proj, 'linear_projection'):
                    #         for param_name, param in expr_proj.linear_projection.named_parameters(recurse=True):
                    #             if not param.requires_grad:
                    #                 param.requires_grad_(True)
                    #                 fixed_count += 1
                    #                 logger.info(f"  ✓ Set requires_grad=True: {name}.expression_projector.linear_projection.{param_name}")
            
            if router_modules_found:
                logger.info(f"✅ Found {len(router_modules_found)} router module(s)")
                # requires_grad 변경하지 않음 (학습에 영향 주지 않도록)
                # if fixed_count > 0:
                #     logger.info(f"✅ Fixed {fixed_count} router parameters: set requires_grad=True")
                # else:
                #     logger.info(f"✅ All router parameters already have requires_grad=True")
            else:
                logger.warning("⚠️ No router modules found in model")
        
        # Trainer가 있으면 optimizer 검증 및 추가
        if trainer is not None:
            # Router 파라미터 강제 설정 및 검증
            validation_result = self._ensure_router_in_optimizer(trainer, model)
            
            if not validation_result['has_routers']:
                logger.warning("⚠️ No router modules found in model (from validation)")
            
            # requires_grad 변경하지 않음 (학습에 영향 주지 않도록)
            # if not validation_result['all_trainable']:
            #     non_trainable = validation_result['non_trainable_params']
            #     logger.warning(f"⚠️ {len(non_trainable)} router parameters still have requires_grad=False - forcing to True...")
            #     
            #     # 실제 파라미터를 찾아서 requires_grad=True로 설정
            #     actual_model = model
            #     if hasattr(model, 'module'):  # DeepSpeed 래핑
            #         actual_model = model.module
            #     
            #     if actual_model is not None:
            #         from models.spectra_model import SPECTRARouter
            #         try:
            #             from models.g3moe_model import G3MoERouter, G3MoEGRINMoE
            #         except ImportError:
            #             G3MoERouter = None
            #             G3MoEGRINMoE = None
            #         
            #         additional_fixed = 0
            #         seen_router_ids = set()  # 이미 처리한 router 인스턴스 추적
            #         
            #         for name, module in actual_model.named_modules():
            #             is_router = False
            #             router_module = None
            #             
            #             if isinstance(module, SPECTRARouter):
            #                 is_router = True
            #                 router_module = module
            #             elif G3MoERouter is not None and isinstance(module, G3MoERouter):
            #                 is_router = True
            #                 router_module = module
            #             elif G3MoEGRINMoE is not None and isinstance(module, G3MoEGRINMoE):
            #                 if hasattr(module, 'router') and isinstance(module.router, G3MoERouter):
            #                     is_router = True
            #                     router_module = module.router
            #                     name = f"{name}.router"
            #             elif hasattr(module, 'load_balancer') and hasattr(module, 'expression_projector'):
            #                 is_router = True
            #                 router_module = module
            #             
            #             if is_router and router_module is not None:
            #                 # 같은 router 인스턴스는 한 번만 처리
            #                 router_id = id(router_module)
            #                 if router_id in seen_router_ids:
            #                     continue
            #                 seen_router_ids.add(router_id)
            #                 # Load balancer 파라미터
            #                 if hasattr(router_module, 'load_balancer'):
            #                     for param_name, param in router_module.load_balancer.named_parameters(recurse=True):
            #                         if not param.requires_grad:
            #                             param.requires_grad_(True)
            #                             additional_fixed += 1
            #                             logger.info(f"  ✓ Set requires_grad=True: {name}.load_balancer.{param_name}")
            #                 
            #                 # Expression projector 파라미터
            #                 if hasattr(router_module, 'expression_projector'):
            #                     expr_proj = router_module.expression_projector
            #                     for param_name, param in expr_proj.named_parameters(recurse=True):
            #                         if not param.requires_grad:
            #                             param.requires_grad_(True)
            #                             additional_fixed += 1
            #                             logger.info(f"  ✓ Set requires_grad=True: {name}.expression_projector.{param_name}")
            #                     
            #                     # linear_projection이 별도로 있는 경우
            #                     if hasattr(expr_proj, 'linear_projection'):
            #                         for param_name, param in expr_proj.linear_projection.named_parameters(recurse=True):
            #                             if not param.requires_grad:
            #                                 param.requires_grad_(True)
            #                                 additional_fixed += 1
            #                                 logger.info(f"  ✓ Set requires_grad=True: {name}.expression_projector.linear_projection.{param_name}")
            #         
            #         if additional_fixed > 0:
            #             logger.info(f"✅ Fixed additional {additional_fixed} router parameters: set requires_grad=True")
            #         
            #         # 재검증
            #         validation_result = self._ensure_router_in_optimizer(trainer, model)
            
            # Optimizer에 없는 파라미터가 있으면 추가
            if not validation_result['all_in_optimizer']:
                missing = validation_result['missing_from_optimizer']
                logger.warning(f"⚠️ {len(missing)} router parameters are not in optimizer - adding to optimizer...")
                
                # 실제 파라미터를 찾아서 optimizer에 추가
                actual_model = model
                if hasattr(model, 'module'):  # DeepSpeed 래핑
                    actual_model = model.module
                
                if actual_model is not None and hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    from models.spectra_model import SPECTRARouter
                    try:
                        from models.g3moe_model import G3MoERouter, G3MoEGRINMoE
                    except ImportError:
                        G3MoERouter = None
                        G3MoEGRINMoE = None
                    
                    optimizer_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
                    missing_params = []
                    seen_router_ids = set()  # 이미 처리한 router 인스턴스 추적
                    
                    for name, module in actual_model.named_modules():
                        is_router = False
                        router_module = None
                        
                        # 1. PEFT ModulesToSaveWrapper 체크 (가장 중요)
                        if ModulesToSaveWrapper is not None and isinstance(module, ModulesToSaveWrapper):
                            active_adapter = getattr(module, "active_adapter", "default")
                            if hasattr(module, "modules_to_save") and active_adapter in module.modules_to_save:
                                inner_module = module.modules_to_save[active_adapter]
                                if isinstance(inner_module, SPECTRARouter):
                                    is_router = True
                                    router_module = inner_module
                                elif G3MoERouter is not None and isinstance(inner_module, G3MoERouter):
                                    is_router = True
                                    router_module = inner_module
                        
                        # 2. SPECTRARouter 체크
                        elif isinstance(module, SPECTRARouter):
                            is_router = True
                            router_module = module
                        # 3. G3MoERouter 직접 체크
                        elif G3MoERouter is not None and isinstance(module, G3MoERouter):
                            is_router = True
                            router_module = module
                        # 4. G3MoEGRINMoE 내부의 router 속성 체크
                        elif G3MoEGRINMoE is not None and isinstance(module, G3MoEGRINMoE):
                            if hasattr(module, 'router'):
                                potential_router = module.router
                                
                                if ModulesToSaveWrapper is not None and isinstance(potential_router, ModulesToSaveWrapper):
                                    active_adapter = getattr(potential_router, "active_adapter", "default")
                                    if hasattr(potential_router, "modules_to_save") and active_adapter in potential_router.modules_to_save:
                                        inner_module = potential_router.modules_to_save[active_adapter]
                                        if isinstance(inner_module, G3MoERouter):
                                            is_router = True
                                            router_module = inner_module
                                            name = f"{name}.router"
                                
                                elif isinstance(potential_router, G3MoERouter):
                                    is_router = True
                                    router_module = potential_router
                                    name = f"{name}.router"
                                    
                        # 5. 일반적인 router 구조 체크 (load_balancer + expression_projector)
                        elif hasattr(module, 'load_balancer') and hasattr(module, 'expression_projector'):
                            is_router = True
                            router_module = module
                        
                        if is_router and router_module is not None:
                            # 같은 router 인스턴스는 한 번만 처리
                            router_id = id(router_module)
                            if router_id in seen_router_ids:
                                continue
                            seen_router_ids.add(router_id)
                            # Load balancer 파라미터
                            if hasattr(router_module, 'load_balancer'):
                                for param_name, param in router_module.load_balancer.named_parameters(recurse=True):
                                    if param.requires_grad and id(param) not in optimizer_param_ids:
                                        missing_params.append(param)
                            
                            # Expression projector 파라미터
                            if hasattr(router_module, 'expression_projector'):
                                expr_proj = router_module.expression_projector
                                for param_name, param in expr_proj.named_parameters(recurse=True):
                                    if param.requires_grad and id(param) not in optimizer_param_ids:
                                        missing_params.append(param)
                                
                                # linear_projection이 별도로 있는 경우
                                if hasattr(expr_proj, 'linear_projection'):
                                    for param_name, param in expr_proj.linear_projection.named_parameters(recurse=True):
                                        if param.requires_grad and id(param) not in optimizer_param_ids:
                                            missing_params.append(param)
                    
                    if missing_params and len(trainer.optimizer.param_groups) > 0:
                        trainer.optimizer.param_groups[0]['params'].extend(missing_params)
                        logger.info(f"  ✓ Added {len(missing_params)} router parameters to optimizer param_groups[0]")
                        
                        # 재검증
                        validation_result = self._ensure_router_in_optimizer(trainer, model)
            
            # 최종 검증 (trainer가 있을 때만)
            if not validation_result['all_trainable']:
                logger.warning(
                    f"⚠️ {len(validation_result['non_trainable_params'])} router parameters still not trainable "
                    f"after attempts to fix. This may cause training issues."
                )
            
            if not validation_result['all_in_optimizer']:
                logger.warning(
                    f"⚠️ {len(validation_result['missing_from_optimizer'])} router parameters still not in optimizer "
                    f"after attempts to add. This may cause training issues."
                )
        
        logger.info("=" * 80)
        logger.info("✅ Router parameter setup complete - all router parameters set to requires_grad=True")
        if trainer is not None:
            logger.info("✅ Router validation passed - all router parameters are trainable and in optimizer")
        logger.info("=" * 80)
        
        # Optimizer에 등록된 파라미터 확인 및 로깅 (train_SPECTRA.py 형식 유지)
        if trainer is not None:
            logger.info("=" * 80)
            logger.info("🔍 Checking parameters registered in optimizer...")
            logger.info("=" * 80)
            
            # Optimizer에서 파라미터 ID 수집 (여러 경로 시도)
            optimizer_param_ids = set()
            optimizer_source = None
            
            # 1. 일반 optimizer 확인
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                try:
                    optimizer_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
                    optimizer_source = "trainer.optimizer"
                    logger.info(f"✅ Found optimizer: trainer.optimizer with {len(optimizer_param_ids)} parameters")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get params from trainer.optimizer: {e}")
            
            # 2. DeepSpeed optimizer 확인
            if not optimizer_param_ids and hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
                if hasattr(trainer.deepspeed, 'optimizer') and trainer.deepspeed.optimizer is not None:
                    try:
                        optimizer_param_ids = {id(p) for group in trainer.deepspeed.optimizer.param_groups for p in group['params']}
                        optimizer_source = "trainer.deepspeed.optimizer"
                        logger.info(f"✅ Found optimizer: trainer.deepspeed.optimizer with {len(optimizer_param_ids)} parameters")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get params from trainer.deepspeed.optimizer: {e}")
            
            if optimizer_param_ids:
                logger.info(f"✅ Total {len(optimizer_param_ids)} parameters in optimizer (source: {optimizer_source})")
                
                # 모델의 모든 파라미터를 순회하면서 optimizer에 등록된 것만 로깅 (train_SPECTRA.py 형식)
                actual_model = model
                if hasattr(model, 'module'):  # DeepSpeed 래핑
                    actual_model = model.module
                
                if actual_model is not None:
                    optimizer_params_logged = 0
                    for name, param in actual_model.named_parameters():
                        # train_SPECTRA.py의 필터링 조건 유지
                        if param.requires_grad and not any([keyword for keyword in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if keyword in name]):
                            # Optimizer에 등록된 파라미터만 로깅
                            if id(param) in optimizer_param_ids:
                                logger.info(f"Trainable Layer: {name} | Shape: {param.shape} | In Optimizer: ✓")
                                optimizer_params_logged += 1
                    
                    logger.info(f"✅ Logged {optimizer_params_logged} trainable parameters that are in optimizer (excluding q/k/v/o/gate/up/down_proj)")
                    
                    # Router 파라미터만 별도로 확인 및 검증
                    from models.spectra_model import SPECTRARouter
                    try:
                        from models.g3moe_model import G3MoERouter, G3MoEGRINMoE
                    except ImportError:
                        G3MoERouter = None
                        G3MoEGRINMoE = None
                    
                    router_params_in_optimizer = 0
                    router_params_not_in_optimizer = 0
                    router_params_list = []
                    router_params_not_in_optimizer_list = []
                    seen_router_ids = set()  # 이미 처리한 router 인스턴스 추적
                    
                    for name, module in actual_model.named_modules():
                        is_router = False
                        router_module = None
                        
                        if isinstance(module, SPECTRARouter):
                            is_router = True
                            router_module = module
                        elif G3MoERouter is not None and isinstance(module, G3MoERouter):
                            is_router = True
                            router_module = module
                        elif G3MoEGRINMoE is not None and isinstance(module, G3MoEGRINMoE):
                            if hasattr(module, 'router') and isinstance(module.router, G3MoERouter):
                                is_router = True
                                router_module = module.router
                                name = f"{name}.router"
                        elif hasattr(module, 'load_balancer') and hasattr(module, 'expression_projector'):
                            is_router = True
                            router_module = module
                        
                        if is_router and router_module is not None:
                            # 같은 router 인스턴스는 한 번만 처리
                            router_id = id(router_module)
                            if router_id in seen_router_ids:
                                continue
                            seen_router_ids.add(router_id)
                            
                            # Router 모듈의 모든 파라미터 확인
                            for param_name, param in router_module.named_parameters(recurse=True):
                                full_name = f"{name}.{param_name}"
                                if param.requires_grad:
                                    param_id = id(param)
                                    if param_id in optimizer_param_ids:
                                        router_params_in_optimizer += 1
                                        router_params_list.append(full_name)
                                        logger.info(f"Trainable Layer: {full_name} | Shape: {param.shape} | In Optimizer: ✓ | param_id={param_id}")
                                    else:
                                        router_params_not_in_optimizer += 1
                                        router_params_not_in_optimizer_list.append(full_name)
                                        logger.warning(f"Trainable Layer: {full_name} | Shape: {param.shape} | In Optimizer: ✗ | param_id={param_id}")
                    
                    # Router 파라미터 검증 결과 요약
                    logger.info("=" * 80)
                    logger.info(f"📊 Router Parameters Optimizer Registration Summary:")
                    logger.info(f"   ✅ In optimizer: {router_params_in_optimizer}")
                    logger.info(f"   ❌ NOT in optimizer: {router_params_not_in_optimizer}")
                    
                    if router_params_in_optimizer > 0:
                        logger.info(f"   Router params in optimizer (first 10):")
                        for param_name in router_params_list[:10]:
                            logger.info(f"     ✓ {param_name}")
                        if len(router_params_list) > 10:
                            logger.info(f"     ... and {len(router_params_list) - 10} more")
                    
                    if router_params_not_in_optimizer > 0:
                        logger.warning(f"   ⚠️ Router params NOT in optimizer (first 10):")
                        for param_name in router_params_not_in_optimizer_list[:10]:
                            logger.warning(f"     ✗ {param_name}")
                        if len(router_params_not_in_optimizer_list) > 10:
                            logger.warning(f"     ... and {len(router_params_not_in_optimizer_list) - 10} more")
                        
                        # CRITICAL: Optimizer에 없는 router 파라미터가 있으면 경고
                        logger.error(f"❌ CRITICAL: {router_params_not_in_optimizer} router parameters are NOT in optimizer!")
                        logger.error(f"   This means these parameters will NOT be updated during training!")
                        logger.error(f"   Router will NOT learn if parameters are not in optimizer!")
                    else:
                        logger.info(f"✅ All router parameters are in optimizer!")
                    
                    logger.info("=" * 80)
            else:
                logger.warning("⚠️ Optimizer not yet initialized - cannot check registered parameters")
                logger.warning("   DeepSpeed lazy initialization: optimizer will be checked at first step (step 0 or 1)")
                logger.warning("   This is normal for DeepSpeed - optimizer initializes after first forward pass")
                self._optimizer_validation_done = False  # 나중에 다시 확인 필요

            logger.info("=" * 80)
        
        logger.info(f"✅ RouterWeightTrackingCallback active - will track router weights every {self.log_every_n_steps} steps")
        if self.check_weight_change:
            logger.info(f"   Weight change checking will start after step {self.check_after_steps}")
        
        # Router forward hook 등록 (실제로 사용된 router 추적)
        if actual_model is not None:
            self._register_router_forward_hooks(actual_model)
        
        return control
    
    def _register_router_forward_hooks(self, model):
        """모든 router 모듈에 forward hook을 등록하여 실제 사용 여부 및 실제 사용되는 weight 추적"""
        from models.spectra_model import SPECTRARouter
        try:
            from models.g3moe_model import G3MoERouter, G3MoEGRINMoE
        except ImportError:
            G3MoERouter = None
            G3MoEGRINMoE = None
        
        router_count = 0
        seen_router_ids = set()  # 이미 처리한 router 인스턴스 추적 (G3MoE에서 global_router와 layers[i].moe.router가 같은 인스턴스)
        
        for name, module in model.named_modules():
            is_router = False
            router_module = None
            wrapper_module = None  # ModulesToSaveWrapper인 경우 wrapper 자체 저장
            active_adapter = "default"
            
            # 1. PEFT ModulesToSaveWrapper 체크 (가장 중요)
            if ModulesToSaveWrapper is not None and isinstance(module, ModulesToSaveWrapper):
                # Wrapper 내부의 실제 학습 모듈('default' 또는 active_adapter) 확인
                active_adapter = getattr(module, "active_adapter", "default")
                if hasattr(module, "modules_to_save") and active_adapter in module.modules_to_save:
                    inner_module = module.modules_to_save[active_adapter]
                    
                    # 내부 모듈이 Router인지 확인
                    if isinstance(inner_module, SPECTRARouter):
                        is_router = True
                        router_module = inner_module
                        wrapper_module = module  # Wrapper 자체 저장
                        logger.debug(f"✅ Found PEFT wrapped router: {name} (adapter: {active_adapter})")
                    elif G3MoERouter is not None and isinstance(inner_module, G3MoERouter):
                        is_router = True
                        router_module = inner_module
                        wrapper_module = module  # Wrapper 자체 저장
                        logger.debug(f"✅ Found PEFT wrapped G3MoE router: {name} (adapter: {active_adapter})")
            
            # 2. SPECTRARouter 체크
            elif isinstance(module, SPECTRARouter):
                is_router = True
                router_module = module
            # 3. G3MoERouter 직접 체크
            elif G3MoERouter is not None and isinstance(module, G3MoERouter):
                is_router = True
                router_module = module
            # 4. G3MoEGRINMoE 내부의 router 속성 체크
            elif G3MoEGRINMoE is not None and isinstance(module, G3MoEGRINMoE):
                # PEFT로 래핑된 router일 수 있음
                if hasattr(module, 'router'):
                    potential_router = module.router
                    
                    # PEFT Wrapper인지 확인
                    if ModulesToSaveWrapper is not None and isinstance(potential_router, ModulesToSaveWrapper):
                        active_adapter = getattr(potential_router, "active_adapter", "default")
                        if hasattr(potential_router, "modules_to_save") and active_adapter in potential_router.modules_to_save:
                            inner_module = potential_router.modules_to_save[active_adapter]
                            if isinstance(inner_module, G3MoERouter):
                                is_router = True
                                router_module = inner_module
                                wrapper_module = potential_router  # Wrapper 자체 저장
                                name = f"{name}.router"
                                logger.debug(f"✅ Found PEFT wrapped nested router in G3MoEGRINMoE: {name}")
                    
                    # 일반 Router인지 확인
                    elif isinstance(potential_router, G3MoERouter):
                        is_router = True
                        router_module = potential_router
                        # 이름을 moe.router로 변경하여 추적
                        name = f"{name}.router"
                        
            # 5. 일반적인 router 구조 체크 (load_balancer + expression_projector)
            elif hasattr(module, 'load_balancer') and hasattr(module, 'expression_projector'):
                is_router = True
                router_module = module
            
            if is_router and router_module is not None:
                # 같은 router 인스턴스는 한 번만 처리 (G3MoE에서 global_router와 layers[i].moe.router가 같은 인스턴스)
                router_id = id(router_module)
                if router_id in seen_router_ids:
                    if self.verbose:
                        logger.debug(f"⏭️ Skipping duplicate router instance: {name} (already processed, router_id={router_id})")
                    continue
                seen_router_ids.add(router_id)
                router_count += 1
                
                # Hook을 등록할 모듈 결정: wrapper가 있으면 wrapper에, 없으면 router_module에
                hook_target_module = wrapper_module if wrapper_module is not None else router_module
                
                # Router 모듈에 대한 forward hook
                def make_router_forward_hook(router_name, inner_router_module, wrapper_mod, adapter_name):
                    def router_forward_hook(hooked_module, input, output):
                        # 현재 step 가져오기 (trainer에서)
                        current_step = None
                        if hasattr(self, '_last_trainer') and self._last_trainer is not None:
                            if hasattr(self._last_trainer, 'state') and self._last_trainer.state is not None:
                                current_step = self._last_trainer.state.global_step
                        
                        if current_step is not None:
                            if current_step not in self._router_forward_tracker:
                                self._router_forward_tracker[current_step] = []
                            
                            # 중복 방지
                            if router_name not in self._router_forward_tracker[current_step]:
                                self._router_forward_tracker[current_step].append(router_name)
                                
                                # Input shape 정보도 기록
                                input_shape = None
                                if input is not None and len(input) > 0:
                                    if isinstance(input[0], torch.Tensor):
                                        input_shape = list(input[0].shape)
                                
                                if self.verbose and current_step <= 5:
                                    logger.info(f"🔍 Router forward called: {router_name} at step {current_step} | input_shape={input_shape}")
                    
                    return router_forward_hook
                
                hook = hook_target_module.register_forward_hook(
                    make_router_forward_hook(name, router_module, wrapper_module, active_adapter)
                )
                self._router_hooks.append((name, hook))
                
                # Expression projector의 linear_projection에 대한 forward hook (실제 사용되는 weight 추적)
                if hasattr(router_module, 'expression_projector'):
                    expr_proj = router_module.expression_projector
                    if hasattr(expr_proj, 'linear_projection'):
                        lin_proj = expr_proj.linear_projection
                        
                        # linear_projection도 ModulesToSaveWrapper로 래핑되어 있을 수 있음
                        lin_proj_wrapper = None
                        lin_proj_inner = None
                        if ModulesToSaveWrapper is not None and isinstance(lin_proj, ModulesToSaveWrapper):
                            lin_proj_adapter = getattr(lin_proj, "active_adapter", "default")
                            if hasattr(lin_proj, "modules_to_save") and lin_proj_adapter in lin_proj.modules_to_save:
                                lin_proj_inner = lin_proj.modules_to_save[lin_proj_adapter]
                                lin_proj_wrapper = lin_proj
                                logger.debug(f"  ✅ Found PEFT wrapped linear_projection: {name}.expression_projector.linear_projection (adapter: {lin_proj_adapter})")
                        
                        # Hook을 등록할 모듈 결정
                        lin_proj_hook_target = lin_proj_wrapper if lin_proj_wrapper is not None else lin_proj
                        
                        def make_linear_projection_hook(router_name, inner_lin_proj, wrapper_lin_proj, adapter_name):
                            def linear_projection_hook(hooked_module, input, output):
                                # 현재 step 가져오기
                                current_step = None
                                if hasattr(self, '_last_trainer') and self._last_trainer is not None:
                                    if hasattr(self._last_trainer, 'state') and self._last_trainer.state is not None:
                                        current_step = self._last_trainer.state.global_step
                                
                                if current_step is not None:
                                    # 실제로 사용되는 weight 추적
                                    # Wrapper인 경우 modules_to_save.default에서 weight 추출
                                    actual_weight = None
                                    if wrapper_lin_proj is not None:
                                        # ModulesToSaveWrapper인 경우, modules_to_save.default의 weight 사용
                                        if hasattr(wrapper_lin_proj, "modules_to_save") and adapter_name in wrapper_lin_proj.modules_to_save:
                                            inner_mod = wrapper_lin_proj.modules_to_save[adapter_name]
                                            if hasattr(inner_mod, 'weight'):
                                                actual_weight = inner_mod.weight
                                    elif inner_lin_proj is not None and hasattr(inner_lin_proj, 'weight'):
                                        actual_weight = inner_lin_proj.weight
                                    elif hasattr(hooked_module, 'weight'):
                                        actual_weight = hooked_module.weight
                                    
                                    if actual_weight is not None:
                                        # router_name을 키로 사용하여 실제 weight 저장
                                        if router_name not in self._actual_router_weights:
                                            self._actual_router_weights[router_name] = {}
                                        
                                        # 현재 step의 weight 저장 (detach & clone)
                                        self._actual_router_weights[router_name][current_step] = actual_weight.detach().clone()
                                        
                                        if self.verbose and current_step <= 5:
                                            param_id = id(actual_weight)
                                            logger.info(f"🔍 Actual weight tracked: {router_name}.expression_projector.linear_projection.weight at step {current_step} | param_id={param_id} | shape={actual_weight.shape} | wrapper={wrapper_lin_proj is not None}")
                            
                            return linear_projection_hook
                        
                        lin_proj_hook = lin_proj_hook_target.register_forward_hook(
                            make_linear_projection_hook(name, lin_proj_inner, lin_proj_wrapper, active_adapter if lin_proj_wrapper is None else getattr(lin_proj_wrapper, "active_adapter", "default"))
                        )
                        self._router_hooks.append((f"{name}.expression_projector.linear_projection", lin_proj_hook))
        
        if router_count > 0:
            logger.info(f"✅ Registered forward hooks on {router_count} router modules - will track which routers are used in forward pass and actual weights")
        else:
            logger.warning("⚠️ No router modules found for forward hook registration")
    
    def on_step_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """
        각 training step 시작 전에 호출
        DeepSpeed lazy initialization 대응: optimizer가 초기화된 후에 검증 수행
        """
        trainer = kwargs.get('trainer')
        if trainer is not None:
            self._last_trainer = trainer
        
        # Optimizer 검증이 아직 안 되었고, 첫 번째 step (0 또는 1)에서 optimizer가 초기화되었는지 확인
        if not self._optimizer_validation_done and state.global_step <= 1:
            # Optimizer가 초기화되었는지 확인
            optimizer_available = False
            optimizer_source = None
            
            if trainer is not None:
                # 1. 일반 optimizer 확인
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    try:
                        param_groups = trainer.optimizer.param_groups
                        if param_groups and len(param_groups) > 0 and len(param_groups[0]['params']) > 0:
                            optimizer_available = True
                            optimizer_source = "trainer.optimizer"
                    except Exception:
                        pass
                
                # 2. DeepSpeed optimizer 확인
                if not optimizer_available and hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
                    if hasattr(trainer.deepspeed, 'optimizer') and trainer.deepspeed.optimizer is not None:
                        try:
                            param_groups = trainer.deepspeed.optimizer.param_groups
                            if param_groups and len(param_groups) > 0 and len(param_groups[0]['params']) > 0:
                                optimizer_available = True
                                optimizer_source = "trainer.deepspeed.optimizer"
                        except Exception:
                            pass
            
            # Optimizer가 초기화되었으면 검증 수행
            if optimizer_available:
                logger.info("=" * 80)
                logger.info(f"🔍 Optimizer initialized! (source: {optimizer_source}, step: {state.global_step})")
                logger.info("   Performing router parameter optimizer registration check...")
                logger.info("=" * 80)
                
                # on_train_begin의 optimizer 검증 로직 재사용
                if trainer is not None and model is not None:
                    # Optimizer에서 파라미터 ID 수집
                    optimizer_param_ids = set()
                    
                    if optimizer_source == "trainer.optimizer":
                        try:
                            optimizer_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get params from trainer.optimizer: {e}")
                    elif optimizer_source == "trainer.deepspeed.optimizer":
                        try:
                            optimizer_param_ids = {id(p) for group in trainer.deepspeed.optimizer.param_groups for p in group['params']}
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get params from trainer.deepspeed.optimizer: {e}")
                    
                    if optimizer_param_ids:
                        logger.info(f"✅ Found {len(optimizer_param_ids)} parameters in optimizer")
                        
                        actual_model = model
                        if hasattr(model, 'module'):  # DeepSpeed 래핑
                            actual_model = model.module
                        
                        if actual_model is not None:
                            from models.spectra_model import SPECTRARouter
                            try:
                                from models.g3moe_model import G3MoERouter
                            except ImportError:
                                G3MoERouter = None
                            
                            router_params_in_optimizer = 0
                            router_params_not_in_optimizer = 0
                            router_params_list = []
                            router_params_not_in_optimizer_list = []
                            
                            try:
                                from models.g3moe_model import G3MoEGRINMoE
                            except ImportError:
                                G3MoEGRINMoE = None
                            
                            seen_router_ids = set()  # 이미 처리한 router 인스턴스 추적
                            
                            for name, module in actual_model.named_modules():
                                is_router = False
                                router_module = None
                                
                                # 1. PEFT ModulesToSaveWrapper 체크 (가장 중요)
                                if ModulesToSaveWrapper is not None and isinstance(module, ModulesToSaveWrapper):
                                    active_adapter = getattr(module, "active_adapter", "default")
                                    if hasattr(module, "modules_to_save") and active_adapter in module.modules_to_save:
                                        inner_module = module.modules_to_save[active_adapter]
                                        if isinstance(inner_module, SPECTRARouter):
                                            is_router = True
                                            router_module = inner_module
                                        elif G3MoERouter is not None and isinstance(inner_module, G3MoERouter):
                                            is_router = True
                                            router_module = inner_module
                                
                                # 2. SPECTRARouter 체크
                                elif isinstance(module, SPECTRARouter):
                                    is_router = True
                                    router_module = module
                                # 3. G3MoERouter 직접 체크
                                elif G3MoERouter is not None and isinstance(module, G3MoERouter):
                                    is_router = True
                                    router_module = module
                                # 4. G3MoEGRINMoE 내부의 router 속성 체크
                                elif G3MoEGRINMoE is not None and isinstance(module, G3MoEGRINMoE):
                                    # PEFT로 래핑된 router일 수 있음
                                    if hasattr(module, 'router'):
                                        potential_router = module.router
                                        
                                        if ModulesToSaveWrapper is not None and isinstance(potential_router, ModulesToSaveWrapper):
                                            active_adapter = getattr(potential_router, "active_adapter", "default")
                                            if hasattr(potential_router, "modules_to_save") and active_adapter in potential_router.modules_to_save:
                                                inner_module = potential_router.modules_to_save[active_adapter]
                                                if isinstance(inner_module, G3MoERouter):
                                                    is_router = True
                                                    router_module = inner_module
                                                    name = f"{name}.router"
                                        
                                        elif isinstance(potential_router, G3MoERouter):
                                            is_router = True
                                            router_module = potential_router
                                            name = f"{name}.router"
                                            
                                # 5. 일반적인 router 구조 체크 (load_balancer + expression_projector)
                                elif hasattr(module, 'load_balancer') and hasattr(module, 'expression_projector'):
                                    is_router = True
                                    router_module = module
                                
                                if is_router and router_module is not None:
                                    # 같은 router 인스턴스는 한 번만 처리
                                    router_id = id(router_module)
                                    if router_id in seen_router_ids:
                                        continue
                                    seen_router_ids.add(router_id)
                                    
                                    for param_name, param in router_module.named_parameters(recurse=True):
                                        full_name = f"{name}.{param_name}"
                                        if param.requires_grad:
                                            param_id = id(param)
                                            if param_id in optimizer_param_ids:
                                                router_params_in_optimizer += 1
                                                router_params_list.append(full_name)
                                                logger.info(f"Trainable Layer: {full_name} | Shape: {param.shape} | In Optimizer: ✓ | param_id={param_id}")
                                            else:
                                                router_params_not_in_optimizer += 1
                                                router_params_not_in_optimizer_list.append(full_name)
                                                logger.warning(f"Trainable Layer: {full_name} | Shape: {param.shape} | In Optimizer: ✗ | param_id={param_id}")
                            
                            # Router 파라미터 검증 결과 요약
                            logger.info("=" * 80)
                            logger.info(f"📊 Router Parameters Optimizer Registration Summary (at step {state.global_step}):")
                            logger.info(f"   ✅ In optimizer: {router_params_in_optimizer}")
                            logger.info(f"   ❌ NOT in optimizer: {router_params_not_in_optimizer}")
                            
                            if router_params_in_optimizer > 0:
                                logger.info(f"   Router params in optimizer (first 10):")
                                for param_name in router_params_list[:10]:
                                    logger.info(f"     ✓ {param_name}")
                                if len(router_params_list) > 10:
                                    logger.info(f"     ... and {len(router_params_list) - 10} more")
                            
                            if router_params_not_in_optimizer > 0:
                                logger.warning(f"   ⚠️ Router params NOT in optimizer (first 10):")
                                for param_name in router_params_not_in_optimizer_list[:10]:
                                    logger.warning(f"     ✗ {param_name}")
                                if len(router_params_not_in_optimizer_list) > 10:
                                    logger.warning(f"     ... and {len(router_params_not_in_optimizer_list) - 10} more")
                                
                                # CRITICAL: Optimizer에 없는 router 파라미터가 있으면 경고
                                logger.error(f"❌ CRITICAL: {router_params_not_in_optimizer} router parameters are NOT in optimizer!")
                                logger.error(f"   This means these parameters will NOT be updated during training!")
                                logger.error(f"   Router will NOT learn if parameters are not in optimizer!")
                            else:
                                logger.info(f"✅ All router parameters are in optimizer!")
                            
                            logger.info("=" * 80)
                
                self._optimizer_validation_done = True
            elif state.global_step == 1:
                # Step 1까지 optimizer가 없으면 경고
                logger.warning(f"⚠️ Optimizer still not initialized at step {state.global_step}")
                logger.warning("   This may indicate an issue with optimizer initialization")
                logger.warning("   Will continue checking, but router parameters may not be in optimizer")
        
        return control
    
    def _ensure_router_in_optimizer(self, trainer, model):
        """
        Router 파라미터가 올바르게 학습 가능한지 검증
        반환값: dict with keys:
            - has_routers: bool
            - all_trainable: bool
            - all_in_optimizer: bool
            - non_trainable_params: list[str]
            - missing_from_optimizer: list[str]
        """
        result = {
            'has_routers': False,
            'all_trainable': False,
            'all_in_optimizer': False,
            'non_trainable_params': [],
            'missing_from_optimizer': []
        }
        
        try:
            from models.spectra_model import SPECTRARouter
            from models.spectra_model import ExpressionProjector
            try:
                from models.g3moe_model import G3MoERouter
            except ImportError:
                G3MoERouter = None
            
            # 모델 추출 (DeepSpeed 래핑 처리)
            actual_model = model
            if hasattr(model, 'module'):  # DeepSpeed 래핑
                actual_model = model.module
            
            if actual_model is None:
                logger.error("❌ Model is None, cannot validate router in optimizer")
                return result
            
            router_params = []
            router_param_names = []
            expression_projector_params = []
            load_balancer_params = []
            seen_param_ids = set()  # 중복 방지
            
            # G3MoEGRINMoE import 추가
            try:
                from models.g3moe_model import G3MoEGRINMoE
            except ImportError:
                G3MoEGRINMoE = None
            
            # 모든 router 파라미터 찾기 (검증용)
            seen_router_ids = set()  # 이미 처리한 router 인스턴스 추적 (G3MoE에서 global_router와 layers[i].moe.router가 같은 인스턴스)
            
            for name, module in actual_model.named_modules():
                is_router = False
                router_module = None
                
                # 1. PEFT ModulesToSaveWrapper 체크 (가장 중요)
                if ModulesToSaveWrapper is not None and isinstance(module, ModulesToSaveWrapper):
                    active_adapter = getattr(module, "active_adapter", "default")
                    if hasattr(module, "modules_to_save") and active_adapter in module.modules_to_save:
                        inner_module = module.modules_to_save[active_adapter]
                        if isinstance(inner_module, SPECTRARouter):
                            is_router = True
                            router_module = inner_module
                        elif G3MoERouter is not None and isinstance(inner_module, G3MoERouter):
                            is_router = True
                            router_module = inner_module
                            
                # 2. SPECTRARouter 체크
                elif isinstance(module, SPECTRARouter):
                    is_router = True
                    router_module = module
                # 3. G3MoERouter 직접 체크
                elif G3MoERouter is not None and isinstance(module, G3MoERouter):
                    is_router = True
                    router_module = module
                # 4. G3MoEGRINMoE 내부의 router 속성 체크
                elif G3MoEGRINMoE is not None and isinstance(module, G3MoEGRINMoE):
                    if hasattr(module, 'router'):
                        potential_router = module.router
                        
                        if ModulesToSaveWrapper is not None and isinstance(potential_router, ModulesToSaveWrapper):
                            active_adapter = getattr(potential_router, "active_adapter", "default")
                            if hasattr(potential_router, "modules_to_save") and active_adapter in potential_router.modules_to_save:
                                inner_module = potential_router.modules_to_save[active_adapter]
                                if isinstance(inner_module, G3MoERouter):
                                    is_router = True
                                    router_module = inner_module
                                    name = f"{name}.router"
                        
                        elif isinstance(potential_router, G3MoERouter):
                            is_router = True
                            router_module = potential_router
                            name = f"{name}.router"
                            
                # 5. 일반적인 router 구조 체크 (load_balancer + expression_projector)
                elif hasattr(module, 'load_balancer') and hasattr(module, 'expression_projector'):
                    is_router = True
                    router_module = module
                
                if is_router and router_module is not None:
                    # 같은 router 인스턴스는 한 번만 처리 (G3MoE에서 global_router와 layers[i].moe.router가 같은 인스턴스)
                    router_id = id(router_module)
                    if router_id in seen_router_ids:
                        continue  # 이미 처리한 router 인스턴스는 스킵
                    seen_router_ids.add(router_id)
                    
                    logger.info(f"✅ Found router: {name} (router_id={router_id})")
                    # Load balancer 파라미터
                    if hasattr(router_module, 'load_balancer'):
                        for param_name, param in router_module.load_balancer.named_parameters(recurse=True):
                            param_id = id(param)
                            if param_id not in seen_param_ids:
                                router_params.append(param)
                                load_balancer_params.append(param)
                                full_name = f"{name}.load_balancer.{param_name}"
                                router_param_names.append(full_name)
                                seen_param_ids.add(param_id)
                    
                    # Expression projector 파라미터
                    if hasattr(router_module, 'expression_projector'):
                        expr_proj = router_module.expression_projector
                        for param_name, param in expr_proj.named_parameters(recurse=True):
                            param_id = id(param)
                            if param_id not in seen_param_ids:
                                router_params.append(param)
                                expression_projector_params.append(param)
                                full_name = f"{name}.expression_projector.{param_name}"
                                router_param_names.append(full_name)
                                seen_param_ids.add(param_id)
                        
                        # linear_projection이 별도로 있는 경우
                        if hasattr(expr_proj, 'linear_projection'):
                            lin_proj = expr_proj.linear_projection
                            
                            # ModulesToSaveWrapper로 래핑되어 있는지 확인
                            if ModulesToSaveWrapper is not None and isinstance(lin_proj, ModulesToSaveWrapper):
                                # Wrapper 내부의 실제 학습 모듈에서 파라미터 추출
                                lin_proj_adapter = getattr(lin_proj, "active_adapter", "default")
                                if hasattr(lin_proj, "modules_to_save") and lin_proj_adapter in lin_proj.modules_to_save:
                                    inner_lin_proj = lin_proj.modules_to_save[lin_proj_adapter]
                                    for param_name, param in inner_lin_proj.named_parameters(recurse=True):
                                        param_id = id(param)
                                        if param_id not in seen_param_ids:
                                            router_params.append(param)
                                            expression_projector_params.append(param)
                                            full_name = f"{name}.expression_projector.linear_projection.{param_name}"
                                            router_param_names.append(full_name)
                                            seen_param_ids.add(param_id)
                            else:
                                # 일반 모듈인 경우
                                for param_name, param in lin_proj.named_parameters(recurse=True):
                                    param_id = id(param)
                                    if param_id not in seen_param_ids:
                                        router_params.append(param)
                                        expression_projector_params.append(param)
                                        full_name = f"{name}.expression_projector.linear_projection.{param_name}"
                                        router_param_names.append(full_name)
                                        seen_param_ids.add(param_id)
            
            if not router_params:
                logger.error("❌ No router parameters found in model")
                return result
            
            result['has_routers'] = True
            logger.info(f"✅ Found {len(router_params)} router parameters")
            logger.info(f"   - Load balancer params: {len(load_balancer_params)}")
            logger.info(f"   - Expression projector params: {len(expression_projector_params)}")
            
            # requires_grad 검증
            non_trainable = []
            for param, param_name in zip(router_params, router_param_names):
                if not param.requires_grad:
                    non_trainable.append(param_name)
            
            if non_trainable:
                result['non_trainable_params'] = non_trainable
                logger.error(f"❌ {len(non_trainable)} router parameters have requires_grad=False")
                for param_name in non_trainable[:5]:
                    logger.error(f"   - {param_name}")
                if len(non_trainable) > 5:
                    logger.error(f"   ... and {len(non_trainable) - 5} more")
            else:
                result['all_trainable'] = True
                logger.info(f"✅ All {len(router_params)} router parameters have requires_grad=True")
            
            # Optimizer 포함 여부 검증 (여러 경로 확인)
            optimizer_param_ids = set()
            optimizer_source = None
            
            # 1. 일반 optimizer 확인
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                try:
                    optimizer_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
                    optimizer_source = "trainer.optimizer"
                    logger.info(f"✅ Found optimizer: trainer.optimizer with {len(optimizer_param_ids)} parameters")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get params from trainer.optimizer: {e}")
            
            # 2. DeepSpeed optimizer 확인
            if not optimizer_param_ids and hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
                if hasattr(trainer.deepspeed, 'optimizer') and trainer.deepspeed.optimizer is not None:
                    try:
                        optimizer_param_ids = {id(p) for group in trainer.deepspeed.optimizer.param_groups for p in group['params']}
                        optimizer_source = "trainer.deepspeed.optimizer"
                        logger.info(f"✅ Found optimizer: trainer.deepspeed.optimizer with {len(optimizer_param_ids)} parameters")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get params from trainer.deepspeed.optimizer: {e}")
            
            if optimizer_param_ids:
                router_param_ids = {id(p) for p in router_params}
                in_optimizer = router_param_ids & optimizer_param_ids
                missing_ids = router_param_ids - optimizer_param_ids
                
                missing_names = [name for param, name in zip(router_params, router_param_names) if id(param) in missing_ids]
                
                logger.info(f"📊 Optimizer registration check (source: {optimizer_source}):")
                logger.info(f"   Total optimizer params: {len(optimizer_param_ids)}")
                logger.info(f"   Router params: {len(router_params)}")
                logger.info(f"   Router params in optimizer: {len(in_optimizer)}/{len(router_params)}")
                
                if missing_names:
                    result['missing_from_optimizer'] = missing_names
                    logger.error(f"❌ {len(missing_names)} router parameters are not in optimizer")
                    for param_name in missing_names[:10]:
                        param = next((p for p, n in zip(router_params, router_param_names) if n == param_name), None)
                        param_id = id(param) if param is not None else "unknown"
                        logger.error(f"   - {param_name} | param_id={param_id}")
                    if len(missing_names) > 10:
                        logger.error(f"   ... and {len(missing_names) - 10} more")
                else:
                    result['all_in_optimizer'] = True
                    logger.info(f"✅ All {len(router_params)} router parameters are in optimizer")
            elif hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
                # DeepSpeed의 경우 requires_grad=True인 파라미터가 자동으로 optimizer에 포함됨
                # 하지만 실제로 optimizer를 확인할 수 있으면 확인하는 것이 좋음
                if hasattr(trainer.deepspeed, 'optimizer') and trainer.deepspeed.optimizer is None:
                    # DeepSpeed optimizer가 아직 초기화되지 않음
                    if result['all_trainable']:
                        result['all_in_optimizer'] = True
                        logger.info("✅ DeepSpeed detected - router params with requires_grad=True will be included automatically (optimizer not yet initialized)")
                    else:
                        logger.error("❌ DeepSpeed detected but router params are not trainable")
                else:
                    # DeepSpeed optimizer가 있지만 param_groups를 가져올 수 없음
                    if result['all_trainable']:
                        result['all_in_optimizer'] = True
                        logger.info("✅ DeepSpeed detected - router params with requires_grad=True will be included automatically")
                    else:
                        logger.error("❌ DeepSpeed detected but router params are not trainable")
            else:
                logger.error("❌ Optimizer not yet initialized - cannot validate router inclusion")
                # Optimizer가 아직 초기화되지 않았으면 requires_grad만 확인
                if result['all_trainable']:
                    result['all_in_optimizer'] = True  # 일단 True로 설정 (초기화 후 확인 필요)
            
            return result
        except Exception as e:
            import traceback
            logger.error(f"❌ Error validating router weights: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return result
    
    def _get_actual_model(self, model):
        """모델에서 실제 모델 추출 (DeepSpeed, DDP 등 처리)"""
        if model is None:
            return None
        
        # DeepSpeed로 감싸진 경우
        if hasattr(model, 'module'):
            return model.module
        
        # 일반 모델
        return model
    
    def on_after_backward(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """
        Backward 이후, optimizer.step() 이전에 호출
        이 시점에서는 gradient가 계산되었지만 weight는 아직 업데이트되지 않음
        Gradient 존재 여부 확인 및 디버깅
        """
        # Trainer 참조 저장 (디버깅용)
        trainer = kwargs.get('trainer')
        if trainer is not None:
            self._last_trainer = trainer
        
        # Gradient 확인 (backward 이후이므로 gradient가 있어야 함)
        # 처음 몇 step과 router weight가 변하지 않는 경우에 자세히 로깅
        should_log_gradients = (
            (self.verbose and state.global_step <= 5) or
            (self.check_weight_change and state.global_step >= self.check_after_steps)
        )
        
        if should_log_gradients:
            log_level = logger.info if state.global_step <= 5 else logger.debug
            log_level(f"RouterWeightTrackingCallback.on_after_backward called at step {state.global_step} (after backward, before optimizer.step())")
            try:
                actual_model = self._get_actual_model(model)
                if actual_model is not None:
                    from models.spectra_model import SPECTRARouter
                    try:
                        from models.g3moe_model import G3MoERouter
                    except ImportError:
                        G3MoERouter = None
                    
                    try:
                        from models.g3moe_model import G3MoEGRINMoE
                    except ImportError:
                        G3MoEGRINMoE = None
                    
                    router_modules = []
                    seen_router_ids = set()  # 이미 처리한 router 인스턴스 추적
                    
                    for name, module in actual_model.named_modules():
                        router_module = None
                        if isinstance(module, SPECTRARouter):
                            router_module = module
                        elif G3MoERouter is not None and isinstance(module, G3MoERouter):
                            router_module = module
                        elif G3MoEGRINMoE is not None and isinstance(module, G3MoEGRINMoE):
                            if hasattr(module, 'router') and isinstance(module.router, G3MoERouter):
                                router_module = module.router
                                name = f"{name}.router"
                        
                        if router_module is not None:
                            # 같은 router 인스턴스는 한 번만 처리
                            router_id = id(router_module)
                            if router_id in seen_router_ids:
                                continue
                            seen_router_ids.add(router_id)
                            router_modules.append((name, router_module))
                    
                    if router_modules:
                        log_level(f"   Found {len(router_modules)} router modules - checking gradients...")
                        total_params_with_grad = 0
                        total_params_requires_grad = 0
                        total_grad_norm = 0.0
                        
                        for router_name, router_module in router_modules[:3]:  # 처음 3개만
                            router_params_with_grad = 0
                            router_params_requires_grad = 0
                            router_grad_norm = 0.0
                            
                            # Expression projector 확인
                            if hasattr(router_module, 'expression_projector'):
                                expr_proj = router_module.expression_projector
                                for param_name, param in expr_proj.named_parameters(recurse=True):
                                    if param.requires_grad:
                                        router_params_requires_grad += 1
                                        total_params_requires_grad += 1
                                        has_grad = param.grad is not None
                                        if has_grad:
                                            router_params_with_grad += 1
                                            total_params_with_grad += 1
                                            grad_norm = param.grad.norm().item()
                                            router_grad_norm += grad_norm
                                            total_grad_norm += grad_norm
                                            if state.global_step <= 5:
                                                log_level(f"     {router_name}.expression_projector.{param_name}: has_grad={has_grad}, grad_norm={grad_norm:.2e}, param_id={id(param)}")
                                            elif not has_grad:
                                                logger.warning(f"     ⚠️ {router_name}.expression_projector.{param_name}: requires_grad=True but grad is None! param_id={id(param)}")
                                
                                # linear_projection 상세 확인 (PEFT 래핑 구조)
                                if hasattr(expr_proj, 'linear_projection'):
                                    lin_proj = expr_proj.linear_projection
                                    
                                    # PEFT ModulesToSaveWrapper 확인
                                    if hasattr(lin_proj, 'original_module') and hasattr(lin_proj, 'modules_to_save'):
                                        orig_module = lin_proj.original_module
                                        modules_to_save = lin_proj.modules_to_save
                                        
                                        # original_module.weight 확인
                                        if hasattr(orig_module, 'weight'):
                                            orig_weight = orig_module.weight
                                            orig_has_grad = orig_weight.grad is not None if hasattr(orig_weight, 'grad') else False
                                            orig_grad_norm = orig_weight.grad.norm().item() if orig_has_grad and orig_weight.grad is not None else 0.0
                                            log_level(f"     {router_name}.expression_projector.linear_projection.original_module.weight: requires_grad={orig_weight.requires_grad}, has_grad={orig_has_grad}, grad_norm={orig_grad_norm:.2e}, param_id={id(orig_weight)}")
                                        
                                        # modules_to_save.default.weight 확인
                                        if hasattr(modules_to_save, 'default') and hasattr(modules_to_save.default, 'weight'):
                                            default_weight = modules_to_save.default.weight
                                            default_has_grad = default_weight.grad is not None if hasattr(default_weight, 'grad') else False
                                            default_grad_norm = default_weight.grad.norm().item() if default_has_grad and default_weight.grad is not None else 0.0
                                            log_level(f"     {router_name}.expression_projector.linear_projection.modules_to_save.default.weight: requires_grad={default_weight.requires_grad}, has_grad={default_has_grad}, grad_norm={default_grad_norm:.2e}, param_id={id(default_weight)}")
                                        
                                        # 직접 접근 weight 확인 (forward에서 실제 사용되는 것)
                                        if hasattr(lin_proj, 'weight'):
                                            direct_weight = lin_proj.weight
                                            direct_has_grad = direct_weight.grad is not None if hasattr(direct_weight, 'grad') else False
                                            direct_grad_norm = direct_weight.grad.norm().item() if direct_has_grad and direct_weight.grad is not None else 0.0
                                            log_level(f"     {router_name}.expression_projector.linear_projection.weight (direct): requires_grad={direct_weight.requires_grad}, has_grad={direct_has_grad}, grad_norm={direct_grad_norm:.2e}, param_id={id(direct_weight)}")
                                            
                                            # 어떤 파라미터와 같은지 확인
                                            if hasattr(orig_module, 'weight') and id(direct_weight) == id(orig_module.weight):
                                                logger.warning(f"     ⚠️ CRITICAL: direct_weight is SAME as original_module.weight! PEFT is using original_module in forward!")
                                            elif hasattr(modules_to_save, 'default') and hasattr(modules_to_save.default, 'weight') and id(direct_weight) == id(modules_to_save.default.weight):
                                                log_level(f"     ✓ direct_weight is SAME as modules_to_save.default.weight (GOOD - using modules_to_save in forward)")
                            
                            # Load balancer 확인
                            if hasattr(router_module, 'load_balancer'):
                                lb_module = router_module.load_balancer
                                for param_name, param in lb_module.named_parameters(recurse=True):
                                    if param.requires_grad:
                                        router_params_requires_grad += 1
                                        total_params_requires_grad += 1
                                        has_grad = param.grad is not None
                                        if has_grad:
                                            router_params_with_grad += 1
                                            total_params_with_grad += 1
                                            grad_norm = param.grad.norm().item()
                                            router_grad_norm += grad_norm
                                            total_grad_norm += grad_norm
                            
                            if router_params_requires_grad > 0:
                                log_level(f"   Router '{router_name}': {router_params_with_grad}/{router_params_requires_grad} params have gradients, total grad_norm={router_grad_norm:.2e}")
                                if router_params_with_grad < router_params_requires_grad:
                                    logger.warning(f"   ⚠️ Router '{router_name}': {router_params_requires_grad - router_params_with_grad} params missing gradients!")
                        
                        # 전체 요약
                        if total_params_requires_grad > 0:
                            log_level(f"   Total router params: {total_params_with_grad}/{total_params_requires_grad} have gradients, total grad_norm={total_grad_norm:.2e}")
                            if total_params_with_grad < total_params_requires_grad:
                                logger.warning(f"   ⚠️ {total_params_requires_grad - total_params_with_grad} router params missing gradients - this may cause router not to learn!")
            except Exception as e:
                logger.warning(f"   Gradient check failed: {e}")
                if self.verbose:
                    import traceback
                    logger.debug(traceback.format_exc())
        
        return control
    
    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """
        각 training step 끝에서 router 가중치 tracking
        NOTE: 이 시점은 backward와 optimizer.step() 이후이므로 weight가 이미 업데이트됨
        """
        # Trainer 참조 저장 (디버깅용)
        trainer = kwargs.get('trainer')
        if trainer is not None:
            self._last_trainer = trainer
        
        # 주기적으로만 tracking (메모리 효율)
        if state.global_step % self.log_every_n_steps == 0:
            
            # 실제 사용되는 weight를 RouterWeightTracker에 전달하기 위해 수정된 추출 함수 사용
            try:
                # 모델 추출
                actual_model = self._get_actual_model(model)
                
                if actual_model is None:
                    if self.verbose and not self._first_step_logged:
                        logger.warning(f"⚠️ RouterWeightTrackingCallback: model is None at step {state.global_step}")
                    return control
                
                # Router 가중치 tracking (optimizer.step() 이후이므로 weight가 업데이트된 상태)
                # Forward hook에서 추적한 실제 사용되는 weight 전달
                step_stats = self.tracker.track_step(
                    model=actual_model,
                    step=state.global_step,
                    global_step=state.global_step,
                    actual_weights_dict=self._actual_router_weights,
                )
                
                # 첫 번째 로깅 시 확인
                if not self._first_step_logged:
                    layers_found = len(step_stats.get('layers', {}))
                    if layers_found > 0:
                        logger.info(f"✅ RouterWeightTrackingCallback: Found {layers_found} router layers at step {state.global_step}")
                    self._first_step_logged = True
                
                # Weight 변화 체크 (check_after_steps 이후부터)
                # Forward hook에서 추적한 실제 사용된 weight를 직접 비교
                actual_weight_changes = None
                if self.check_weight_change and state.global_step >= self.check_after_steps:
                    should_stop, actual_weight_changes = self._check_actual_weight_changes(state.global_step, model=actual_model, trainer=trainer)
                    if should_stop:
                        control.should_training_stop = True
                        control.should_epoch_stop = True
                        logger.error("🛑 Training stopped due to router weights not changing!")
                        return control
                
                # Wandb에 로깅 (선택적)
                if hasattr(args, 'report_to') and args.report_to and 'wandb' in args.report_to:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb_logs = {}
                            
                            # Forward hook에서 추적한 실제 weight 변화 로깅
                            if actual_weight_changes:
                                for router_name, change_info in actual_weight_changes.items():
                                    for metric_name, metric_value in change_info.items():
                                        if isinstance(metric_value, (int, float)):
                                            wandb_logs[f"router_weight_actual/{router_name}/{metric_name}"] = metric_value
                            
                            # 기존 통계 로깅
                            for layer_key, layer_data in step_stats.get('layers', {}).items():
                                if 'load_balancer' in layer_data:
                                    lb_stats = layer_data['load_balancer']
                                    for stat_name, stat_value in lb_stats.items():
                                        if isinstance(stat_value, (int, float)):
                                            wandb_logs[f"router_weight/{layer_key}/load_balancer/{stat_name}"] = stat_value
                                
                                if 'expression_projector' in layer_data:
                                    expr_stats = layer_data['expression_projector']
                                    for stat_name, stat_value in expr_stats.items():
                                        if isinstance(stat_value, (int, float)):
                                            wandb_logs[f"router_weight/{layer_key}/expression_projector/{stat_name}"] = stat_value
                                
                                if 'load_balancer_changes' in layer_data:
                                    changes = layer_data['load_balancer_changes']
                                    for change_name, change_value in changes.items():
                                        if isinstance(change_value, (int, float)):
                                            wandb_logs[f"router_weight/{layer_key}/load_balancer_change/{change_name}"] = change_value
                                
                                if 'expression_projector_changes' in layer_data:
                                    changes = layer_data['expression_projector_changes']
                                    for change_name, change_value in changes.items():
                                        if isinstance(change_value, (int, float)):
                                            wandb_logs[f"router_weight/{layer_key}/expression_projector_change/{change_name}"] = change_value
                            
                            # Bias balancing monitoring metrics
                            try:
                                from models.spectra_model import SPECTRARouter
                                
                                all_bias_magnitudes = []
                                all_bias_changes = []
                                total_router_count = 0
                                
                                for name, module in actual_model.named_modules():
                                    if isinstance(module, SPECTRARouter) and hasattr(module, 'expert_bias'):
                                        total_router_count += 1
                                        
                                        # Expert bias statistics
                                        expert_bias = module.expert_bias
                                        if expert_bias.numel() > 0:
                                            bias_l2_norm = torch.norm(expert_bias, p=2).item()
                                            bias_mean = expert_bias.mean().item()
                                            bias_std = expert_bias.std().item()
                                            bias_max = expert_bias.max().item()
                                            bias_min = expert_bias.min().item()
                                            
                                            all_bias_magnitudes.append(bias_l2_norm)
                                            
                                            # Per-expert bias values (first 10 experts only to avoid clutter)
                                            for expert_idx in range(min(10, expert_bias.numel())):
                                                wandb_logs[f"bias/expert_bias_{expert_idx}"] = expert_bias[expert_idx].item()
                                            
                                            # Bias change from previous step
                                            if hasattr(module, 'prev_expert_bias') and module.prev_expert_bias.numel() > 0:
                                                bias_change = expert_bias - module.prev_expert_bias
                                                bias_change_norm = torch.norm(bias_change, p=2).item()
                                                all_bias_changes.append(bias_change_norm)
                                                
                                                # Update prev_expert_bias for next step
                                                module.prev_expert_bias.copy_(expert_bias.detach())
                                            else:
                                                # Initialize prev_expert_bias
                                                if not hasattr(module, 'prev_expert_bias'):
                                                    module.register_buffer("prev_expert_bias", expert_bias.detach().clone())
                                                else:
                                                    module.prev_expert_bias.copy_(expert_bias.detach())
                                            
                                            # Expert usage statistics
                                            if hasattr(module, 'last_current_load') and module.last_current_load is not None:
                                                current_load = module.last_current_load
                                                total_tokens = current_load.sum().item()
                                                
                                                if total_tokens > 0:
                                                    usage_distribution = (current_load / total_tokens).cpu().numpy()
                                                    target_per_expert = 1.0 / float(module.num_experts)
                                                    
                                                    # Usage deviation from uniform
                                                    deviation = usage_distribution - target_per_expert
                                                    usage_deviation = float(np.linalg.norm(deviation))
                                                    
                                                    # Coefficient of variation
                                                    usage_mean = usage_distribution.mean()
                                                    usage_std = usage_distribution.std()
                                                    usage_cv = float(usage_std / (usage_mean + 1e-8))
                                                    
                                                    # Unused experts count
                                                    unused_count = int((current_load == 0).sum().item())
                                                    
                                                    # Max/min usage ratio
                                                    max_usage = float(current_load.max().item())
                                                    min_usage = float(current_load.min().item())
                                                    avg_usage = float(current_load.mean().item())
                                                    max_usage_ratio = float(max_usage / (avg_usage + 1e-8))
                                                    min_usage_ratio = float(min_usage / (avg_usage + 1e-8)) if min_usage > 0 else 0.0
                                                    
                                                    # Per-expert usage (first 10 experts only)
                                                    for expert_idx in range(min(10, current_load.numel())):
                                                        wandb_logs[f"usage/expert_usage_{expert_idx}"] = float(current_load[expert_idx].item())
                                                    
                                                    # Aggregate usage metrics
                                                    wandb_logs[f"usage/expert_usage_deviation"] = usage_deviation
                                                    wandb_logs[f"usage/expert_usage_cv"] = usage_cv
                                                    wandb_logs[f"usage/unused_experts_count"] = unused_count
                                                    wandb_logs[f"usage/max_usage_ratio"] = max_usage_ratio
                                                    wandb_logs[f"usage/min_usage_ratio"] = min_usage_ratio
                                            
                                            # Router-specific bias statistics
                                            router_layer_key = name.replace('.', '_')
                                            wandb_logs[f"router/{router_layer_key}/bias/expert_bias_l2_norm"] = bias_l2_norm
                                            wandb_logs[f"router/{router_layer_key}/bias/expert_bias_mean"] = bias_mean
                                            wandb_logs[f"router/{router_layer_key}/bias/expert_bias_std"] = bias_std
                                            wandb_logs[f"router/{router_layer_key}/bias/expert_bias_max"] = bias_max
                                            wandb_logs[f"router/{router_layer_key}/bias/expert_bias_min"] = bias_min
                                
                                # Aggregate bias metrics across all routers
                                if all_bias_magnitudes:
                                    wandb_logs["bias/expert_bias_l2_norm"] = np.mean(all_bias_magnitudes)
                                    wandb_logs["bias/expert_bias_mean"] = np.mean(all_bias_magnitudes)
                                
                                if all_bias_changes:
                                    wandb_logs["bias/expert_bias_change_norm"] = np.mean(all_bias_changes)
                                
                            except Exception as e:
                                logger.debug(f"Failed to log bias balancing metrics: {e}")
                            
                            if wandb_logs:
                                wandb.log(wandb_logs, step=state.global_step, commit=False)
                    except ImportError:
                        pass
                    except Exception as e:
                        logger.debug(f"Failed to log to wandb at step {state.global_step}: {e}")
            
            except Exception as e:
                logger.error(f"❌ Failed to track router weights at step {state.global_step}: {e}")
                if self.verbose:
                    import traceback
                    logger.error(traceback.format_exc())
        
        return control
    
    def _check_actual_weight_changes(self, step: int, model=None, trainer=None):
        """
        Forward hook에서 추적한 실제 사용된 weight의 변화를 직접 체크
        
        Returns:
            (should_stop: bool, weight_changes: dict) - 학습 중단 여부와 weight 변화 정보
        """
        # 현재 step에서 실제 사용된 weight 확인 (forward hook에서 추적한 것)
        current_step_weights = {}
        for router_name, step_weights in self._actual_router_weights.items():
            if step in step_weights:
                current_step_weights[router_name] = step_weights[step]
        
        if not current_step_weights:
            return False, None
        
        # 이전 step의 forward hook weight와 비교
        if not self._prev_actual_weights:
            self._prev_actual_weights = {k: v.detach().clone() for k, v in current_step_weights.items()}
            return False, None
        
        # Weight 변화 계산
        all_changes_zero = True
        max_change = 0.0
        change_details = []
        weight_changes = {}  # wandb 로깅용
        
        for router_name, current_weight in current_step_weights.items():
            if router_name not in self._prev_actual_weights:
                continue
            
            prev_weight = self._prev_actual_weights[router_name]
            
            if prev_weight.shape != current_weight.shape:
                continue
            
            try:
                diff = current_weight - prev_weight
                diff_norm = float(torch.norm(diff).item())
                diff_mean = float(diff.mean().item())
                diff_max = float(diff.abs().max().item())
                diff_std = float(diff.std().item())
                
                max_change = max(max_change, diff_norm)
                change_details.append(f"{router_name}: diff_norm={diff_norm:.2e}, diff_mean={diff_mean:.2e}, diff_max={diff_max:.2e}")
                
                # wandb 로깅용 저장
                weight_changes[router_name] = {
                    'diff_norm': diff_norm,
                    'diff_mean': diff_mean,
                    'diff_max': diff_max,
                    'diff_std': diff_std,
                }
                
                if diff_norm >= self.min_change_threshold:
                    all_changes_zero = False
            except Exception as e:
                logger.debug(f"Failed to compute weight change for {router_name}: {e}")
                continue
        
        # 변화가 없으면 학습 중단
        if all_changes_zero:
            error_msg = (
                f"\n{'='*80}\n"
                f"❌ ROUTER WEIGHT CHANGE CHECK FAILED at step {step}\n"
                f"{'='*80}\n"
                f"All router weight changes are below threshold ({self.min_change_threshold:.2e})\n"
                f"This means the router is NOT LEARNING!\n"
                f"\nMax change observed: {max_change:.2e}\n"
                f"Threshold: {self.min_change_threshold:.2e}\n"
                f"\nChange details (actual weights used in forward, comparing step {step-1} vs {step}):\n"
            )
            for detail in change_details[:20]:
                error_msg += f"  {detail}\n"
            error_msg += f"\n{'='*80}\n"
            
            logger.error(error_msg)
            
            # 디버깅: optimizer에 등록되어 있는지 확인
            if trainer is not None:
                logger.error("🔍 Checking if router weights are in optimizer...")
                try:
                    optimizer_param_ids = set()
                    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                        optimizer_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
                    elif hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None:
                        if hasattr(trainer.deepspeed, 'optimizer') and trainer.deepspeed.optimizer is not None:
                            optimizer_param_ids = {id(p) for group in trainer.deepspeed.optimizer.param_groups for p in group['params']}
                    
                    for router_name, current_weight in current_step_weights.items():
                        weight_param_id = id(current_weight)
                        in_optimizer = weight_param_id in optimizer_param_ids
                        logger.error(f"   {router_name}: weight param_id={weight_param_id}, in_optimizer={in_optimizer}")
                except Exception as e:
                    logger.debug(f"Failed to check optimizer: {e}")
            
            return True, weight_changes
        
        # 현재 step의 forward hook weight를 이전 weight로 업데이트
        self._prev_actual_weights = {k: v.detach().clone() for k, v in current_step_weights.items()}
        return False, weight_changes
    
    def _check_weight_changes(self, step_stats: Dict[str, Any], step: int, model=None, trainer=None) -> bool:
        """
        Weight 변화를 체크하고, 변화가 없으면 학습 중단
        
        Args:
            step_stats: 현재 step의 통계
            step: 현재 step 번호
            model: 모델 인스턴스 (디버깅용)
            trainer: Trainer 인스턴스 (디버깅용)
            
        Returns:
            True if should stop training, False otherwise
        """
        layers = step_stats.get('layers', {})
        
        if not layers:
            if self.verbose:
                logger.warning(f"⚠️ No router layers found for weight change check at step {step}")
            return False
        
        all_changes_zero = True
        max_change = 0.0
        change_details = []
        has_change_data = False  # 변화 데이터가 있는지 확인
        
        for layer_key, layer_data in layers.items():
            # Load balancer 변화 체크
            if 'load_balancer_changes' in layer_data:
                has_change_data = True
                lb_changes = layer_data['load_balancer_changes']
                for change_name, change_value in lb_changes.items():
                    if isinstance(change_value, (int, float)):
                        abs_change = abs(change_value)
                        max_change = max(max_change, abs_change)
                        change_details.append(f"{layer_key}.load_balancer.{change_name}={change_value:.2e}")
                        
                        if abs_change >= self.min_change_threshold:
                            all_changes_zero = False
            
            # Expression projector 변화 체크
            if 'expression_projector_changes' in layer_data:
                has_change_data = True
                expr_changes = layer_data['expression_projector_changes']
                for change_name, change_value in expr_changes.items():
                    if isinstance(change_value, (int, float)):
                        abs_change = abs(change_value)
                        max_change = max(max_change, abs_change)
                        change_details.append(f"{layer_key}.expression_projector.{change_name}={change_value:.2e}")
                        
                        if abs_change >= self.min_change_threshold:
                            all_changes_zero = False
        
        # 변화 데이터가 없으면 (첫 step 등) 체크 스킵
        if not has_change_data:
            if self.verbose:
                logger.debug(f"⚠️ No change data available at step {step} (first step?), skipping check")
            return False
        
        # 변화 데이터가 있는데 모두 0이면 학습 중단
        if all_changes_zero:
            error_msg = (
                f"\n{'='*80}\n"
                f"❌ ROUTER WEIGHT CHANGE CHECK FAILED at step {step}\n"
                f"{'='*80}\n"
                f"All router weight changes are below threshold ({self.min_change_threshold:.2e})\n"
                f"This means the router is NOT LEARNING!\n"
                f"\nMax change observed: {max_change:.2e}\n"
                f"Threshold: {self.min_change_threshold:.2e}\n"
                f"\nChange details (first 10):\n"
            )
            for detail in change_details[:10]:
                error_msg += f"  {detail}\n"
            error_msg += f"\n{'='*80}\n"
            
            logger.error(error_msg)
            
            # 디버깅: expression_projector 파라미터 상태 확인 및 forward에서 실제 사용되는 파라미터 추적
            logger.error("🔍 Debugging expression_projector parameters:")
            try:
                from models.spectra_model import SPECTRARouter
                
                # 실제 모델에서 router 찾기 (전달받은 model/trainer 우선 사용)
                actual_model = None
                debug_trainer = None
                
                # 1. 함수 인자로 전달받은 model/trainer 우선 사용
                if model is not None:
                    actual_model = self._get_actual_model(model)
                    debug_trainer = trainer
                # 2. 저장된 trainer 참조 사용
                elif hasattr(self, '_last_trainer') and self._last_trainer is not None:
                    debug_trainer = self._last_trainer
                    actual_model = self._get_actual_model(debug_trainer.model if hasattr(debug_trainer, 'model') else None)
                # 3. trainer 인자로 전달받은 경우
                elif trainer is not None:
                    debug_trainer = trainer
                    actual_model = self._get_actual_model(trainer.model if hasattr(trainer, 'model') else None)
                
                if actual_model is None:
                    logger.error("   ⚠️ Cannot access model for debugging (model and trainer not available)")
                    logger.error("   This may indicate a problem with callback integration")
                
                if actual_model is not None:
                    try:
                        from models.g3moe_model import G3MoERouter, G3MoEGRINMoE
                    except ImportError:
                        G3MoERouter = None
                        G3MoEGRINMoE = None
                    
                    router_modules = []
                    seen_router_ids = set()  # 이미 처리한 router 인스턴스 추적
                    
                    for name, module in actual_model.named_modules():
                        router_module = None
                        if isinstance(module, SPECTRARouter):
                            router_module = module
                        elif G3MoERouter is not None and isinstance(module, G3MoERouter):
                            router_module = module
                        elif G3MoEGRINMoE is not None and isinstance(module, G3MoEGRINMoE):
                            if hasattr(module, 'router') and isinstance(module.router, G3MoERouter):
                                router_module = module.router
                                name = f"{name}.router"
                        
                        if router_module is not None:
                            # 같은 router 인스턴스는 한 번만 처리
                            router_id = id(router_module)
                            if router_id in seen_router_ids:
                                continue
                            seen_router_ids.add(router_id)
                            router_modules.append((name, router_module))
                    
                    logger.error(f"   Found {len(router_modules)} router modules")
                    
                    for router_name, router_module in router_modules[:3]:  # 처음 3개만
                        if hasattr(router_module, 'expression_projector'):
                            expr_proj = router_module.expression_projector
                            logger.error(f"   Router {router_name}:")
                            
                            # 파라미터 상태 확인
                            expr_params = list(expr_proj.named_parameters(recurse=True))
                            logger.error(f"     Expression projector params: {len(expr_params)}")
                            
                            # 모든 파라미터 확인 (original_module 포함)
                            for param_name, param in expr_params:
                                has_grad = param.grad is not None if hasattr(param, 'grad') else False
                                grad_norm = param.grad.norm().item() if has_grad and param.grad is not None else 0.0
                                logger.error(f"       {param_name}: shape={param.shape}, requires_grad={param.requires_grad}, has_grad={has_grad}, grad_norm={grad_norm:.2e}, param_id={id(param)}")
                            
                            # linear_projection 확인 - PEFT 래핑 구조 분석
                            if hasattr(expr_proj, 'linear_projection'):
                                lin_proj = expr_proj.linear_projection
                                logger.error(f"     linear_projection module type: {type(lin_proj)}")
                                logger.error(f"     linear_projection attributes: {[attr for attr in dir(lin_proj) if not attr.startswith('_')]}")
                                
                                # PEFT ModulesToSaveWrapper 확인
                                has_original_module = hasattr(lin_proj, 'original_module')
                                has_modules_to_save = hasattr(lin_proj, 'modules_to_save')
                                
                                logger.error(f"     Has original_module: {has_original_module}")
                                logger.error(f"     Has modules_to_save: {has_modules_to_save}")
                                
                                if has_original_module:
                                    orig_module = lin_proj.original_module
                                    logger.error(f"     original_module type: {type(orig_module)}")
                                    if hasattr(orig_module, 'weight'):
                                        orig_weight = orig_module.weight
                                        orig_weight_grad = orig_weight.grad is not None if hasattr(orig_weight, 'grad') else False
                                        orig_weight_grad_norm = orig_weight.grad.norm().item() if orig_weight_grad and orig_weight.grad is not None else 0.0
                                        logger.error(f"     original_module.weight: shape={orig_weight.shape}, requires_grad={orig_weight.requires_grad}, has_grad={orig_weight_grad}, grad_norm={orig_weight_grad_norm:.2e}, param_id={id(orig_weight)}")
                                
                                if has_modules_to_save:
                                    modules_to_save = lin_proj.modules_to_save
                                    logger.error(f"     modules_to_save type: {type(modules_to_save)}")
                                    logger.error(f"     modules_to_save keys: {list(modules_to_save.keys()) if hasattr(modules_to_save, 'keys') else 'N/A'}")
                                    
                                    if hasattr(modules_to_save, 'default'):
                                        default_module = modules_to_save.default
                                        logger.error(f"     modules_to_save.default type: {type(default_module)}")
                                        if hasattr(default_module, 'weight'):
                                            default_weight = default_module.weight
                                            default_weight_grad = default_weight.grad is not None if hasattr(default_weight, 'grad') else False
                                            default_weight_grad_norm = default_weight.grad.norm().item() if default_weight_grad and default_weight.grad is not None else 0.0
                                            logger.error(f"     modules_to_save.default.weight: shape={default_weight.shape}, requires_grad={default_weight.requires_grad}, has_grad={default_weight_grad}, grad_norm={default_weight_grad_norm:.2e}, param_id={id(default_weight)}")
                                
                                # 직접 weight 속성 확인 (PEFT가 forward에서 사용하는 것)
                                if hasattr(lin_proj, 'weight'):
                                    direct_weight = lin_proj.weight
                                    direct_weight_grad = direct_weight.grad is not None if hasattr(direct_weight, 'grad') else False
                                    direct_weight_grad_norm = direct_weight.grad.norm().item() if direct_weight_grad and direct_weight.grad is not None else 0.0
                                    logger.error(f"     linear_projection.weight (direct access): shape={direct_weight.shape}, requires_grad={direct_weight.requires_grad}, has_grad={direct_weight_grad}, grad_norm={direct_weight_grad_norm:.2e}, param_id={id(direct_weight)}")
                                    
                                    # 어떤 파라미터와 같은지 확인
                                    if has_original_module and hasattr(lin_proj.original_module, 'weight'):
                                        if id(direct_weight) == id(lin_proj.original_module.weight):
                                            logger.error(f"     ⚠️ CRITICAL: direct_weight is SAME as original_module.weight!")
                                            logger.error(f"     This confirms PEFT is using original_module in forward pass!")
                                    if has_modules_to_save and hasattr(modules_to_save, 'default') and hasattr(modules_to_save.default, 'weight'):
                                        if id(direct_weight) == id(modules_to_save.default.weight):
                                            logger.error(f"     ✓ direct_weight is SAME as modules_to_save.default.weight (GOOD!)")
                                
                                # Forward hook을 통한 실제 사용 파라미터 추적
                                def forward_hook(module, input, output):
                                    if hasattr(module, 'weight'):
                                        weight = module.weight
                                        logger.error(f"     🔍 FORWARD HOOK: linear_projection forward called with weight param_id={id(weight)}, requires_grad={weight.requires_grad}")
                                
                                # Hook 등록 (다음 forward에서 확인)
                                if not hasattr(lin_proj, '_debug_hook_registered'):
                                    lin_proj.register_forward_hook(forward_hook)
                                    lin_proj._debug_hook_registered = True
                                    logger.error(f"     ✓ Registered forward hook for linear_projection (will log on next forward pass)")
                            
                            # Optimizer 확인
                            if debug_trainer is not None:
                                # DeepSpeed 케이스 확인
                                if hasattr(debug_trainer, 'deepspeed') and debug_trainer.deepspeed is not None:
                                    if hasattr(debug_trainer.deepspeed, 'optimizer') and debug_trainer.deepspeed.optimizer is not None:
                                        ds_optimizer = debug_trainer.deepspeed.optimizer
                                        if hasattr(ds_optimizer, 'param_groups'):
                                            optimizer_param_ids = {id(p) for group in ds_optimizer.param_groups for p in group['params']}
                                            expr_param_ids = {id(p) for _, p in expr_params}
                                            in_optimizer = expr_param_ids & optimizer_param_ids
                                            logger.error(f"     Params in DeepSpeed optimizer: {len(in_optimizer)}/{len(expr_param_ids)}")
                                            
                                            # 어떤 파라미터가 optimizer에 있는지 상세 확인
                                            for param_name, param in expr_params:
                                                param_id = id(param)
                                                in_opt = param_id in optimizer_param_ids
                                                logger.error(f"       {param_name}: in_optimizer={in_opt}, param_id={param_id}")
                                        else:
                                            logger.error(f"     ⚠️ DeepSpeed optimizer has no param_groups")
                                    else:
                                        logger.error(f"     ⚠️ DeepSpeed optimizer not yet initialized")
                                # 일반 optimizer 케이스
                                elif hasattr(debug_trainer, 'optimizer') and debug_trainer.optimizer is not None:
                                    optimizer_param_ids = {id(p) for group in debug_trainer.optimizer.param_groups for p in group['params']}
                                    expr_param_ids = {id(p) for _, p in expr_params}
                                    in_optimizer = expr_param_ids & optimizer_param_ids
                                    logger.error(f"     Params in optimizer: {len(in_optimizer)}/{len(expr_param_ids)}")
                                    
                                    # 어떤 파라미터가 optimizer에 있는지 상세 확인
                                    for param_name, param in expr_params:
                                        param_id = id(param)
                                        in_opt = param_id in optimizer_param_ids
                                        logger.error(f"       {param_name}: in_optimizer={in_opt}, param_id={param_id}")
                                else:
                                    logger.error(f"     ⚠️ Optimizer not available in trainer")
                            else:
                                logger.error(f"     ⚠️ Cannot check optimizer (trainer not available)")
                else:
                    logger.error("   ⚠️ Cannot access model for debugging")
                                    
            except Exception as debug_e:
                logger.error(f"   Debug logging failed: {debug_e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # 학습 중단을 위해 True 반환
            return True
        else:
            if self.verbose and step % (self.log_every_n_steps * 10) == 0:
                logger.info(f"✅ Router weight changes OK at step {step}: max_change={max_change:.2e}")
            return False
    
    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Training 종료 시 최종 요약 저장 및 hook 제거"""
        # Forward hook 제거
        for router_name, hook in self._router_hooks:
            try:
                hook.remove()
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove forward hook for {router_name}: {e}")
        
        if self._router_hooks:
            logger.info(f"✅ Removed {len(self._router_hooks)} router forward hooks")
        
        # Router forward 사용 통계 요약
        if self._router_forward_tracker:
            all_used_routers = set()
            for step, routers in self._router_forward_tracker.items():
                all_used_routers.update(routers)
            
            logger.info("=" * 80)
            logger.info(f"📊 Router Forward Pass Summary:")
            logger.info(f"   Total steps with router usage: {len(self._router_forward_tracker)}")
            logger.info(f"   Unique routers used: {len(all_used_routers)}")
            logger.info(f"   Routers used during training:")
            for router_name in sorted(all_used_routers):
                usage_count = sum(1 for routers in self._router_forward_tracker.values() if router_name in routers)
                logger.info(f"     ✓ {router_name} (used in {usage_count} steps)")
            logger.info("=" * 80)
        
        try:
            # 최종 요약 저장
            summary = self.tracker.save_summary()
            if self.verbose:
                logger.info(f"✅ Router weight tracking summary saved: {summary}")
                logger.info(f"   Total steps tracked: {summary.get('total_steps', 0)}")
                logger.info(f"   Layers tracked: {summary.get('layers_tracked', [])}")
        except Exception as e:
            logger.error(f"❌ Failed to save router weight summary: {e}")
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
        
        return control
    
    def on_save(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Checkpoint 저장 시 router 가중치도 함께 저장"""
        try:
            # Checkpoint 디렉토리에 router 가중치 요약 저장
            if state.is_world_process_zero:
                checkpoint_dir = os.path.join(
                    args.output_dir,
                    f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
                )
                if os.path.exists(checkpoint_dir):
                    summary_file = os.path.join(checkpoint_dir, "router_weight_summary.json")
                    summary = self.tracker.save_summary(summary_file)
                    if self.verbose:
                        logger.info(f"✅ Router weight summary saved to checkpoint: {summary_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save router weight summary to checkpoint: {e}")
            if self.verbose:
                import traceback
                logger.error(traceback.format_exc())
        
        return control
