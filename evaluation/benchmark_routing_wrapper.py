#!/usr/bin/env python3
"""
벤치마크 실행을 위한 라우팅 메트릭 수집 래퍼

모델을 래핑하여 벤치마크 실행 중 라우팅 메트릭을 수집합니다.

양자화/압축 모델 지원:
- AWQ: 가중치만 양자화하므로 forward hook 정상 작동
- LLMCompression: 모델 구조가 유지되면 작동 가능
- 기타 양자화: 가중치만 변경하는 경우 일반적으로 작동
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.benchmark_routing_collector import BenchmarkRoutingCollector


class RoutingMetricsModelWrapper:
    """
    모델을 래핑하여 벤치마크 실행 중 라우팅 메트릭을 수집
    
    모델의 forward pass에 hook을 등록하고,
    각 태스크별로 라우팅 메트릭을 수집합니다.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_experts: int,
        router_dim: int = 128,
        capacity_factor: float = 1.25,
    ):
        self.model = model
        self.num_experts = num_experts
        self.router_dim = router_dim
        self.capacity_factor = capacity_factor
        
        # 라우팅 메트릭 수집기
        self.collector = BenchmarkRoutingCollector(
            num_experts=num_experts,
            router_dim=router_dim,
            capacity_factor=capacity_factor
        )
        
        # Hook 등록 여부
        self.hooks_registered = False
        
    def register_hooks(self):
        """모델에 forward hook 등록"""
        if not self.hooks_registered:
            num_layers = self.collector.register_hooks(self.model)
            self.hooks_registered = True
            return num_layers
        return 0
    
    def remove_hooks(self):
        """Hook 제거"""
        if self.hooks_registered:
            self.collector.remove_hooks()
            self.hooks_registered = False
    
    def reset_for_task(self):
        """새 태스크 시작 전 초기화"""
        self.collector.reset()
    
    def get_task_metrics(self) -> Dict[str, Any]:
        """
        현재 태스크의 라우팅 메트릭 반환
        
        Returns:
            Dict[str, Any]: 논문용 핵심 메트릭
        """
        return self.collector.get_task_metrics()
    
    def __call__(self, *args, **kwargs):
        """모델 forward pass 호출"""
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """모델의 다른 속성 접근"""
        return getattr(self.model, name)


def create_routing_wrapper(
    model: nn.Module,
    num_experts: Optional[int] = None,
    router_dim: int = 128,
    capacity_factor: float = 1.25,
) -> RoutingMetricsModelWrapper:
    """
    모델에서 num_experts를 자동 감지하여 래퍼 생성
    
    Args:
        model: 래핑할 모델
        num_experts: Expert 수 (None이면 자동 감지)
        router_dim: Router dimension
        capacity_factor: Capacity factor
    
    Returns:
        RoutingMetricsModelWrapper 인스턴스
    """
    # num_experts 자동 감지
    if num_experts is None:
        num_experts = _detect_num_experts(model)
        if num_experts is None:
            raise ValueError("Could not detect num_experts from model. Please specify explicitly.")
    
    return RoutingMetricsModelWrapper(
        model=model,
        num_experts=num_experts,
        router_dim=router_dim,
        capacity_factor=capacity_factor,
    )


def _detect_num_experts(model: nn.Module) -> Optional[int]:
    """모델에서 num_experts 자동 감지"""
    # Seqorth 모델
    if hasattr(model, 'config'):
        config = model.config
        # Seqorth config 구조 확인
        if hasattr(config, 'text_config'):
            text_config = config.text_config
            if hasattr(text_config, 'n_routed_experts'):
                return text_config.n_routed_experts
            if hasattr(text_config, 'num_experts'):
                return text_config.num_experts
        
        # 일반 config
        if hasattr(config, 'n_routed_experts'):
            return config.n_routed_experts
        if hasattr(config, 'num_experts'):
            return config.num_experts
    
    # MoE 레이어에서 직접 확인
    for name, module in model.named_modules():
        if hasattr(module, 'num_experts'):
            return module.num_experts
        if hasattr(module, 'n_routed_experts'):
            return module.n_routed_experts
        if hasattr(module, 'experts'):
            experts = module.experts
            if isinstance(experts, (list, nn.ModuleList)):
                return len(experts)
            elif hasattr(experts, '__len__'):
                return len(experts)
        if hasattr(module, 'router'):
            router = module.router
            if hasattr(router, 'num_experts'):
                return router.num_experts
    
    return None
