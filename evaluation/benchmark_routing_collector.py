#!/usr/bin/env python3
"""
라우팅 메트릭 수집 로직

벤치마크 실행 중 forward hook을 통해 라우팅 정보를 추출하고
SeqorthAnalyzer로 메트릭을 계산합니다.

양자화/압축 모델 지원:
- AWQ (Activation-aware Weight Quantization): 가중치만 양자화하므로 forward hook과 
  중간 activation 접근 가능. 라우팅 메트릭 수집 정상 작동.
- LLMCompression: 구조가 유지되면 forward hook 작동 가능. 
  다만 모델 구조가 변경된 경우 수동 조정 필요할 수 있음.
- 기타 양자화 기법 (GPTQ, QAT 등): 가중치만 양자화하는 경우 일반적으로 작동.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.seqorth_analysis import SeqorthAnalyzer


class BenchmarkRoutingCollector:
    """
    벤치마크 실행 중 라우팅 메트릭을 수집하는 클래스
    
    Forward hook을 등록하여 MoE 레이어에서 라우팅 정보를 추출하고
    SeqorthAnalyzer로 메트릭을 계산합니다.
    """
    
    def __init__(self, num_experts: int, router_dim: int = 128, capacity_factor: float = 1.25):
        self.num_experts = num_experts
        self.router_dim = router_dim
        self.capacity_factor = capacity_factor
        
        # SeqorthAnalyzer 인스턴스 (태스크별로 reset)
        self.analyzer = SeqorthAnalyzer(
            num_experts=num_experts,
            router_dim=router_dim,
            capacity_factor=capacity_factor
        )
        
        # Hook 관리
        self.hooks = []
        self.routing_data_buffer = []  # Forward pass 중 수집된 라우팅 정보
        
    def register_hooks(self, model: nn.Module):
        """모델의 MoE 레이어에 forward hook 등록"""
        self.remove_hooks()  # 기존 hook 제거
        
        # MoE 레이어 찾기
        moe_layers = self._find_moe_layers(model)
        
        for layer_name, module in moe_layers:
            hook = module.register_forward_hook(
                self._create_hook_fn(layer_name)
            )
            self.hooks.append((layer_name, hook))
        
        return len(self.hooks)
    
    def remove_hooks(self):
        """등록된 모든 hook 제거"""
        for layer_name, hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _find_moe_layers(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """모델에서 MoE 레이어 찾기"""
        moe_layers = []
        
        for name, module in model.named_modules():
            # Seqorth 모델: SeqorthMoE 블록 찾기
            if 'SeqorthMoE' in type(module).__name__ or 'SeqorthBlock' in type(module).__name__:
                moe_layers.append((name, module))
            # 외부 모델: MoE 관련 키워드로 찾기
            elif any(keyword in name.lower() for keyword in ['moe', 'expert', 'router']):
                # MoE 관련 속성 확인
                has_experts = (
                    hasattr(module, 'experts') or
                    hasattr(module, 'num_experts') or
                    hasattr(module, 'router') or
                    hasattr(module, 'gate')
                )
                if has_experts:
                    moe_layers.append((name, module))
        
        return moe_layers
    
    def _create_hook_fn(self, layer_name: str):
        """Forward hook 함수 생성"""
        def hook_fn(module, input, output):
            try:
                routing_info = self._extract_routing_info(module, input, output)
                if routing_info:
                    routing_info['layer_name'] = layer_name
                    self.routing_data_buffer.append(routing_info)
            except Exception as e:
                # Hook 에러가 벤치마크 실행을 방해하지 않도록
                pass
        
        return hook_fn
    
    def _extract_routing_info(self, module, input, output) -> Optional[Dict[str, Any]]:
        """
        모듈에서 라우팅 정보 추출
        
        moe_monitoring_callback.py의 _extract_routing_info 로직 재사용
        
        양자화/압축 모델 지원:
        - AWQ, LLMCompression 등 양자화 모델에서도 작동
        - Forward hook은 모델 구조가 유지되면 정상 작동
        - 중간 activation (routing logits, expert selections) 접근 가능
        """
        routing_info = {}
        
        # ===== Seqorth 모델: Router에서 직접 추출 =====
        router = getattr(module, 'router', None)
        if router is not None:
            # selected_experts 추출
            if hasattr(router, 'last_selected_experts') and router.last_selected_experts is not None:
                selected_experts = router.last_selected_experts
                if selected_experts.dim() == 2:
                    routing_info['selected_experts'] = selected_experts
                else:
                    routing_info['selected_experts'] = selected_experts.flatten()
            
            # routing_weights 추출
            if hasattr(router, 'last_routing_weights') and router.last_routing_weights is not None:
                routing_weights = router.last_routing_weights
                if routing_weights.dim() == 2:
                    routing_info['routing_weights'] = routing_weights
                else:
                    routing_info['routing_weights'] = routing_weights.flatten()
            
            # num_experts
            if hasattr(router, 'num_experts'):
                routing_info['num_experts'] = router.num_experts
            
            # routing_logits (Seqorth의 경우)
            if hasattr(router, 'last_routing_logits') and router.last_routing_logits is not None:
                routing_info['routing_logits'] = router.last_routing_logits
            
            # expression_logits
            if hasattr(router, 'last_expression_logits') and router.last_expression_logits is not None:
                routing_info['expression_logits'] = router.last_expression_logits
            
            # cosine_similarities
            if hasattr(router, 'last_cosine_similarities') and router.last_cosine_similarities is not None:
                routing_info['cosine_similarities'] = router.last_cosine_similarities
            
            # speciality_loss
            if hasattr(router, 'last_speciality_loss') and router.last_speciality_loss is not None:
                val = router.last_speciality_loss
                if torch.is_tensor(val):
                    routing_info['speciality_penalty'] = val.item() if val.numel() == 1 else val.mean().item()
                else:
                    routing_info['speciality_penalty'] = val
            
            # expression_loss
            if hasattr(router, 'last_expression_reg_loss') and router.last_expression_reg_loss is not None:
                val = router.last_expression_reg_loss
                if torch.is_tensor(val):
                    routing_info['expression_loss'] = val.item() if val.numel() == 1 else val.mean().item()
                else:
                    routing_info['expression_loss'] = val
        
        # ===== 외부 모델: Output에서 추출 =====
        if not routing_info and isinstance(output, tuple):
            # DeepSeek-V3 등 외부 모델의 경우 output 구조에서 추출 시도
            # 일반적으로 (hidden_states, routing_info) 형태
            if len(output) >= 2:
                routing_info_tuple = output[-1]
                if isinstance(routing_info_tuple, (dict, tuple)):
                    # 라우팅 정보가 포함된 경우 처리
                    pass  # 외부 모델은 구조가 다양하므로 기본 추출만 수행
        
        # ===== 모듈 레벨에서 추출 (fallback) =====
        if 'selected_experts' not in routing_info and hasattr(module, 'last_selected_experts'):
            selected_experts = module.last_selected_experts
            if selected_experts is not None:
                routing_info['selected_experts'] = selected_experts.flatten() if selected_experts.dim() > 1 else selected_experts
        
        return routing_info if routing_info else None
    
    def process_routing_data(self):
        """
        수집된 라우팅 데이터를 SeqorthAnalyzer로 처리하여 메트릭 계산
        
        Returns:
            Dict[str, Any]: 계산된 메트릭
        """
        if not self.routing_data_buffer:
            return {}
        
        # 각 라우팅 step 처리
        for routing_info in self.routing_data_buffer:
            try:
                # SeqorthAnalyzer에 필요한 정보 추출
                selected_experts = routing_info.get('selected_experts')
                routing_weights = routing_info.get('routing_weights')
                routing_logits = routing_info.get('routing_logits')
                expression_logits = routing_info.get('expression_logits')
                cosine_similarities = routing_info.get('cosine_similarities')
                speciality_penalty = routing_info.get('speciality_penalty', 0.0)
                expression_loss = routing_info.get('expression_loss', 0.0)
                
                # 텐서 형태 정규화
                if selected_experts is not None and torch.is_tensor(selected_experts):
                    if selected_experts.dim() == 1:
                        # [batch*seq] -> [batch*seq, 1] (top_k=1 가정)
                        selected_experts = selected_experts.unsqueeze(1)
                    elif selected_experts.dim() > 2:
                        selected_experts = selected_experts.view(-1, selected_experts.shape[-1])
                
                if routing_weights is not None and torch.is_tensor(routing_weights):
                    if routing_weights.dim() == 1:
                        routing_weights = routing_weights.unsqueeze(1)
                    elif routing_weights.dim() > 2:
                        routing_weights = routing_weights.view(-1, routing_weights.shape[-1])
                
                # routing_logits 정규화 (Seqorth의 경우)
                if routing_logits is None and expression_logits is not None:
                    # expression_logits를 routing_logits로 사용 (fallback)
                    routing_logits = expression_logits
                
                if routing_logits is not None and torch.is_tensor(routing_logits):
                    # [batch*seq, num_experts, router_dim] 형태로 정규화
                    if routing_logits.dim() == 2:
                        # [batch*seq, num_experts*router_dim] -> [batch*seq, num_experts, router_dim]
                        batch_seq = routing_logits.shape[0]
                        routing_logits = routing_logits.view(batch_seq, self.num_experts, self.router_dim)
                    elif routing_logits.dim() == 3:
                        # 이미 올바른 형태
                        pass
                    elif routing_logits.dim() == 4:
                        # [batch, seq, num_experts, router_dim] -> [batch*seq, num_experts, router_dim]
                        batch, seq, num_exp, router_dim = routing_logits.shape
                        routing_logits = routing_logits.view(batch * seq, num_exp, router_dim)
                
                if expression_logits is not None and torch.is_tensor(expression_logits):
                    # routing_logits와 동일한 형태로 정규화
                    if expression_logits.dim() == 2:
                        batch_seq = expression_logits.shape[0]
                        expression_logits = expression_logits.view(batch_seq, self.num_experts, self.router_dim)
                    elif expression_logits.dim() == 4:
                        batch, seq, num_exp, router_dim = expression_logits.shape
                        expression_logits = expression_logits.view(batch * seq, num_exp, router_dim)
                
                if cosine_similarities is not None and torch.is_tensor(cosine_similarities):
                    # [batch, seq, num_experts] 형태로 정규화
                    if cosine_similarities.dim() == 2:
                        # [batch*seq, num_experts] -> [batch, seq, num_experts]
                        batch_seq = cosine_similarities.shape[0]
                        # batch와 seq를 추정 (정확하지 않을 수 있음)
                        seq_len = 512  # 기본값
                        batch_size = batch_seq // seq_len
                        if batch_size * seq_len == batch_seq:
                            cosine_similarities = cosine_similarities.view(batch_size, seq_len, -1)
                    elif cosine_similarities.dim() == 3:
                        # 이미 올바른 형태
                        pass
                
                # SeqorthAnalyzer로 메트릭 계산
                if selected_experts is not None and routing_weights is not None:
                    # 최소한의 정보가 있으면 분석 시도
                    if routing_logits is None:
                        # routing_logits가 없으면 더미 생성 (메트릭 일부만 계산 가능)
                        batch_seq = selected_experts.shape[0]
                        routing_logits = torch.zeros(
                            batch_seq, self.num_experts, self.router_dim,
                            device=selected_experts.device,
                            dtype=selected_experts.dtype
                        )
                    
                    if expression_logits is None:
                        expression_logits = routing_logits
                    
                    if cosine_similarities is None:
                        batch_size = selected_experts.shape[0] // 512 if selected_experts.shape[0] > 512 else 1
                        seq_len = selected_experts.shape[0] // batch_size
                        cosine_similarities = torch.zeros(
                            batch_size, seq_len, self.num_experts,
                            device=selected_experts.device,
                            dtype=selected_experts.dtype
                        )
                    
                    self.analyzer.analyze_routing_step(
                        routing_logits=routing_logits,
                        expression_logits=expression_logits,
                        selected_experts=selected_experts,
                        routing_weights=routing_weights,
                        speciality_penalty=speciality_penalty,
                        cosine_similarities=cosine_similarities,
                        expression_loss=expression_loss,
                    )
            except Exception as e:
                # 개별 step 에러가 전체 수집을 방해하지 않도록
                continue
        
        # 집계 메트릭 반환
        aggregated = self.analyzer.get_aggregated_metrics()
        paper_metrics = self.analyzer.get_paper_metrics_summary()
        
        return {
            'aggregated_metrics': aggregated,
            'paper_metrics': paper_metrics,
        }
    
    def reset(self):
        """태스크별로 초기화"""
        self.analyzer.reset()
        self.routing_data_buffer.clear()
    
    def get_task_metrics(self) -> Dict[str, Any]:
        """
        현재 태스크의 라우팅 메트릭 반환
        
        Returns:
            Dict[str, Any]: 논문용 메트릭 요약
        """
        metrics = self.process_routing_data()
        
        # 논문용 핵심 메트릭만 추출
        paper_metrics = metrics.get('paper_metrics', {})
        aggregated = metrics.get('aggregated_metrics', {})
        
        # 핵심 메트릭 정리
        core_metrics = {
            # Load Balancing
            'coefficient_of_variation': aggregated.get('final_load_balancing_cv'),
            'maxvio': aggregated.get('final_maxvio'),
            'gini_coefficient': aggregated.get('final_gini_coefficient'),
            'min_max_expert_load_ratio': aggregated.get('final_min_max_expert_load_ratio'),
            
            # Expert Specialization
            'expert_entropy': aggregated.get('final_expert_activation_entropy'),
            'expert_overlap': aggregated.get('final_expert_overlap_mean'),
            'gram_orthogonality': aggregated.get('final_gram_matrix_orthogonality'),
            
            # Routing Quality
            'routing_entropy': aggregated.get('final_routing_entropy'),
            'routing_consistency': aggregated.get('final_sequential_routing_consistency'),
        }
        
        return core_metrics
