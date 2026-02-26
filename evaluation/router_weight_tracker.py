# coding=utf-8
"""
Router Weight Tracker for Seqorth MoE

이 모듈은 seqorth 모델의 router 가중치를 step별로 tracking하고 분석합니다.
Router 가중치는 다음 두 부분으로 구성됩니다:
1. load_balancer (GRU): sequential routing을 위한 가중치
2. expression_projector: orthogonal expression projection을 위한 가중치
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
import os
from pathlib import Path


def extract_router_weights(model: nn.Module, actual_weights_dict: Optional[Dict[str, Dict[int, torch.Tensor]]] = None) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    모델에서 모든 router 가중치를 추출합니다.
    
    Args:
        model: Seqorth MoE 모델
        
    Returns:
        Dictionary with structure:
        {
            'layer_{idx}': {
                'load_balancer': {
                    'weight_ih': tensor,  # Input-to-hidden weights
                    'weight_hh': tensor,  # Hidden-to-hidden weights
                },
                'expression_projector': {
                    'weight': tensor,  # Expression projection weights
                    'bias': tensor,    # Expression projection bias (if exists)
                }
            }
        }
    """
    router_weights = {}
    
    # 모델 구조에 따라 layers 찾기
    layers = None
    for attr_name in ['layers', 'h', 'block', 'decoder_layers']:
        if hasattr(model, attr_name):
            candidate = getattr(model, attr_name)
            if isinstance(candidate, (nn.ModuleList, list)):
                layers = candidate
                break
    
    # Nested access (model.model.layers, etc.)
    if layers is None:
        for attr_name in ['model', 'language_model', 'decoder', 'transformer']:
            if hasattr(model, attr_name):
                submodel = getattr(model, attr_name)
                for name in ['layers', 'h', 'block', 'decoder_layers']:
                    if hasattr(submodel, name):
                        candidate = getattr(submodel, name)
                        if isinstance(candidate, (nn.ModuleList, list)):
                            layers = candidate
                            break
                if layers is not None:
                    break
    
    if layers is None:
        return router_weights
    
    # 각 레이어에서 router 가중치 추출
    for layer_idx, layer in enumerate(layers):
        # MLP/MoE 블록 찾기
        moe_block = None
        for attr_name in ['mlp', 'feed_forward', 'ffn', 'ffw', 'moe']:
            if hasattr(layer, attr_name):
                candidate = getattr(layer, attr_name)
                # SeqorthBlock인지 확인
                if hasattr(candidate, 'router') and hasattr(candidate.router, 'load_balancer'):
                    moe_block = candidate
                    break
        
        if moe_block is None:
            continue
        
        router = moe_block.router
        layer_key = f'layer_{layer_idx}'
        router_weights[layer_key] = {}
        
        # 1. Load Balancer (GRU) 가중치 추출
        if hasattr(router, 'load_balancer'):
            load_balancer = router.load_balancer
            if isinstance(load_balancer, nn.GRU):
                # GRU의 가중치 추출
                gru_weights = {}
                for name, param in load_balancer.named_parameters():
                    # GRU 파라미터 이름: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
                    # param이 None이거나 빈 텐서인 경우 건너뛰기
                    if param is None or param.numel() == 0:
                        continue
                    
                    try:
                        if 'weight_ih' in name:
                            gru_weights['weight_ih'] = param.detach().clone()
                        elif 'weight_hh' in name:
                            gru_weights['weight_hh'] = param.detach().clone()
                        elif 'bias_ih' in name:
                            gru_weights['bias_ih'] = param.detach().clone()
                        elif 'bias_hh' in name:
                            gru_weights['bias_hh'] = param.detach().clone()
                    except Exception:
                        # 가중치 추출 실패 시 건너뛰기
                        continue
                
                # gru_weights가 비어있지 않은 경우만 저장
                if gru_weights:
                    router_weights[layer_key]['load_balancer'] = gru_weights
        
        # 2. Expression Projector 가중치 추출
        if hasattr(router, 'expression_projector'):
            expr_proj = router.expression_projector
            expr_weights = {}
            
            # CRITICAL: PEFT 래핑된 경우 실제로 forward에서 사용되는 weight를 추출
            # modules_to_save.default.linear_projection.weight를 직접 접근
            if hasattr(expr_proj, 'linear_projection'):
                lin_proj = expr_proj.linear_projection
                
                # PEFT ModulesToSaveWrapper 확인 - modules_to_save.default.linear_projection.weight 직접 접근
                if hasattr(lin_proj, 'modules_to_save') and hasattr(lin_proj.modules_to_save, 'default'):
                    default_module = lin_proj.modules_to_save.default
                    # default_module이 ExpressionProjector인 경우
                    if hasattr(default_module, 'linear_projection'):
                        default_lin_proj = default_module.linear_projection
                        if hasattr(default_lin_proj, 'weight'):
                            actual_weight = default_lin_proj.weight
                            if actual_weight is not None and actual_weight.numel() > 0:
                                try:
                                    expr_weights['weight'] = actual_weight.detach().clone()
                                except Exception:
                                    pass
                        if hasattr(default_lin_proj, 'bias') and default_lin_proj.bias is not None:
                            actual_bias = default_lin_proj.bias
                            if actual_bias is not None and actual_bias.numel() > 0:
                                try:
                                    expr_weights['bias'] = actual_bias.detach().clone()
                                except Exception:
                                    pass
                    # default_module이 직접 Linear인 경우
                    elif isinstance(default_module, nn.Linear):
                        if hasattr(default_module, 'weight'):
                            actual_weight = default_module.weight
                            if actual_weight is not None and actual_weight.numel() > 0:
                                try:
                                    expr_weights['weight'] = actual_weight.detach().clone()
                                except Exception:
                                    pass
                        if hasattr(default_module, 'bias') and default_module.bias is not None:
                            actual_bias = default_module.bias
                            if actual_bias is not None and actual_bias.numel() > 0:
                                try:
                                    expr_weights['bias'] = actual_bias.detach().clone()
                                except Exception:
                                    pass
                
                # Forward hook에서 추적한 실제 weight가 있으면 우선 사용 (더 정확함)
                if actual_weights_dict is not None:
                    # router_name을 찾기 위해 모델에서 router의 전체 경로 찾기
                    router_full_name = None
                    for router_name, router_module in model.named_modules():
                        if router_module is router:
                            router_full_name = router_name
                            break
                    
                    if router_full_name is not None and router_full_name in actual_weights_dict:
                        # 가장 최근 step의 weight 사용
                        if actual_weights_dict[router_full_name]:
                            latest_step = max(actual_weights_dict[router_full_name].keys())
                            actual_weight = actual_weights_dict[router_full_name][latest_step]
                            if actual_weight is not None and actual_weight.numel() > 0:
                                try:
                                    expr_weights['weight'] = actual_weight.detach().clone()
                                except Exception:
                                    pass
                
                # 위 방법들이 모두 실패한 경우 lin_proj.weight 직접 접근 시도
                if 'weight' not in expr_weights and hasattr(lin_proj, 'weight'):
                    actual_weight = lin_proj.weight
                    if actual_weight is not None and actual_weight.numel() > 0:
                        try:
                            expr_weights['weight'] = actual_weight.detach().clone()
                        except Exception:
                            pass
                    if 'bias' not in expr_weights and hasattr(lin_proj, 'bias') and lin_proj.bias is not None:
                        actual_bias = lin_proj.bias
                        if actual_bias is not None and actual_bias.numel() > 0:
                            try:
                                expr_weights['bias'] = actual_bias.detach().clone()
                            except Exception:
                                pass
                
                # PEFT 래핑이 아니거나 위 방법들이 실패한 경우 기존 로직 사용
                if 'weight' not in expr_weights or 'bias' not in expr_weights:
                    # ExpressionProjector의 구조에 따라 가중치 추출
                    for name, param in expr_proj.named_parameters():
                        # param이 None이거나 빈 텐서인 경우 건너뛰기
                        if param is None or param.numel() == 0:
                            continue
                        
                        try:
                            if 'weight' in name and 'weight' not in expr_weights:
                                expr_weights['weight'] = param.detach().clone()
                            elif 'bias' in name and 'bias' not in expr_weights:
                                expr_weights['bias'] = param.detach().clone()
                        except Exception:
                            # 가중치 추출 실패 시 건너뛰기
                            continue
            else:
                # linear_projection이 없는 경우 기존 로직 사용
                for name, param in expr_proj.named_parameters():
                    # param이 None이거나 빈 텐서인 경우 건너뛰기
                    if param is None or param.numel() == 0:
                        continue
                    
                    try:
                        if 'weight' in name and 'weight' not in expr_weights:
                            expr_weights['weight'] = param.detach().clone()
                        elif 'bias' in name and 'bias' not in expr_weights:
                            expr_weights['bias'] = param.detach().clone()
                    except Exception:
                        # 가중치 추출 실패 시 건너뛰기
                        continue
            
            # ExpressionProjector가 Linear layer를 포함하는 경우
            if hasattr(expr_proj, 'projection'):
                if isinstance(expr_proj.projection, nn.Linear):
                    try:
                        if expr_proj.projection.weight is not None and expr_proj.projection.weight.numel() > 0:
                            expr_weights['projection_weight'] = expr_proj.projection.weight.detach().clone()
                        if expr_proj.projection.bias is not None and expr_proj.projection.bias.numel() > 0:
                            expr_weights['projection_bias'] = expr_proj.projection.bias.detach().clone()
                    except Exception:
                        pass
            
            # ExpressionProjector가 여러 Linear layer를 포함하는 경우
            if hasattr(expr_proj, 'projections'):
                if isinstance(expr_proj.projections, nn.ModuleList):
                    for i, proj in enumerate(expr_proj.projections):
                        if isinstance(proj, nn.Linear):
                            try:
                                if proj.weight is not None and proj.weight.numel() > 0:
                                    expr_weights[f'projection_{i}_weight'] = proj.weight.detach().clone()
                                if proj.bias is not None and proj.bias.numel() > 0:
                                    expr_weights[f'projection_{i}_bias'] = proj.bias.detach().clone()
                            except Exception:
                                continue
            
            # expr_weights가 비어있지 않은 경우만 저장
            if expr_weights:
                router_weights[layer_key]['expression_projector'] = expr_weights
    
    return router_weights


def compute_weight_statistics(weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    가중치 텐서의 통계량을 계산합니다.
    
    Args:
        weights: Dictionary of weight tensors
        
    Returns:
        Dictionary with statistics (mean, std, min, max, norm) for each weight
    """
    stats = {}
    
    for name, tensor in weights.items():
        if tensor is None:
            continue
        
        # 빈 텐서 체크 (numel() == 0)
        if tensor.numel() == 0:
            # 빈 텐서인 경우 기본값 반환
            stats[f'{name}_mean'] = 0.0
            stats[f'{name}_std'] = 0.0
            stats[f'{name}_min'] = 0.0
            stats[f'{name}_max'] = 0.0
            stats[f'{name}_norm'] = 0.0
            stats[f'{name}_shape'] = list(tensor.shape)
            continue
        
        # 정상적인 텐서인 경우 통계 계산
        try:
            stats[f'{name}_mean'] = float(tensor.mean().item())
            stats[f'{name}_std'] = float(tensor.std().item())
            stats[f'{name}_min'] = float(tensor.min().item())
            stats[f'{name}_max'] = float(tensor.max().item())
            stats[f'{name}_norm'] = float(torch.norm(tensor).item())
            stats[f'{name}_shape'] = list(tensor.shape)
        except Exception as e:
            # 통계 계산 실패 시 기본값 반환
            stats[f'{name}_mean'] = 0.0
            stats[f'{name}_std'] = 0.0
            stats[f'{name}_min'] = 0.0
            stats[f'{name}_max'] = 0.0
            stats[f'{name}_norm'] = 0.0
            stats[f'{name}_shape'] = list(tensor.shape) if hasattr(tensor, 'shape') else []
    
    return stats


def compute_weight_change(
    weights_before: Dict[str, torch.Tensor],
    weights_after: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    두 시점의 가중치 간 변화를 계산합니다.
    
    Args:
        weights_before: 이전 step의 가중치
        weights_after: 현재 step의 가중치
        
    Returns:
        Dictionary with change metrics (diff_norm, diff_mean, diff_max) for each weight
    """
    changes = {}
    
    for name in set(weights_before.keys()) & set(weights_after.keys()):
        w_before = weights_before[name]
        w_after = weights_after[name]
        
        # None 체크
        if w_before is None or w_after is None:
            continue
        
        # Shape 불일치 체크
        if w_before.shape != w_after.shape:
            continue
        
        # 빈 텐서 체크
        if w_before.numel() == 0 or w_after.numel() == 0:
            # 빈 텐서인 경우 기본값 반환
            changes[f'{name}_diff_norm'] = 0.0
            changes[f'{name}_diff_mean'] = 0.0
            changes[f'{name}_diff_max'] = 0.0
            changes[f'{name}_diff_std'] = 0.0
            continue
        
        try:
            diff = w_after - w_before
            
            # diff가 빈 텐서인 경우 체크
            if diff.numel() == 0:
                changes[f'{name}_diff_norm'] = 0.0
                changes[f'{name}_diff_mean'] = 0.0
                changes[f'{name}_diff_max'] = 0.0
                changes[f'{name}_diff_std'] = 0.0
                continue
            
            changes[f'{name}_diff_norm'] = float(torch.norm(diff).item())
            changes[f'{name}_diff_mean'] = float(diff.mean().item())
            changes[f'{name}_diff_max'] = float(diff.abs().max().item())
            changes[f'{name}_diff_std'] = float(diff.std().item())
        except Exception as e:
            # 계산 실패 시 기본값 반환
            changes[f'{name}_diff_norm'] = 0.0
            changes[f'{name}_diff_mean'] = 0.0
            changes[f'{name}_diff_max'] = 0.0
            changes[f'{name}_diff_std'] = 0.0
    
    return changes


class RouterWeightTracker:
    """
    Router 가중치를 step별로 tracking하는 클래스
    """
    
    def __init__(
        self,
        save_dir: str = "./router_weight_logs",
        save_every_n_steps: int = 100,
        save_full_weights: bool = False,  # True면 전체 가중치 저장 (메모리 많이 사용)
        max_history: int = 1000,  # 메모리에 유지할 최대 step 수
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_steps = save_every_n_steps
        self.save_full_weights = save_full_weights
        self.max_history = max_history
        
        # Step별 가중치 히스토리 (메모리 효율을 위해 통계만 저장)
        self.weight_history: List[Dict[str, Any]] = []
        self.weight_snapshots: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}  # 특정 step의 전체 가중치
        
        # 이전 step의 가중치 (변화 계산용)
        self.prev_weights: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        
    def track_step(
        self,
        model: nn.Module,
        step: int,
        global_step: Optional[int] = None,
        actual_weights_dict: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        """
        현재 step의 router 가중치를 tracking합니다.
        
        Args:
            model: Seqorth MoE 모델
            step: 현재 step 번호
            global_step: 전역 step 번호 (None이면 step 사용)
            actual_weights_dict: Forward hook에서 추적한 실제 사용되는 weight 딕셔너리
                                {router_name: {step: weight_tensor}}
            
        Returns:
            Tracking 결과 딕셔너리
        """
        if global_step is None:
            global_step = step
        
        # Router 가중치 추출 (forward hook에서 추적한 실제 weight 전달)
        current_weights = extract_router_weights(model, actual_weights_dict=actual_weights_dict)
        
        # 각 레이어별 통계 계산
        step_stats = {
            'step': step,
            'global_step': global_step,
            'layers': {}
        }
        
        for layer_key, layer_weights in current_weights.items():
            layer_stats = {}
            
            # Load balancer 통계
            if 'load_balancer' in layer_weights:
                lb_stats = compute_weight_statistics(layer_weights['load_balancer'])
                layer_stats['load_balancer'] = lb_stats
            
            # Expression projector 통계
            if 'expression_projector' in layer_weights:
                expr_stats = compute_weight_statistics(layer_weights['expression_projector'])
                layer_stats['expression_projector'] = expr_stats
            
            # 이전 step과의 변화 계산
            # CRITICAL: prev_weights는 이전 step의 weight (optimizer.step() 이후, 즉 이전 step의 최종 weight)
            # current_weights는 현재 step의 weight (optimizer.step() 이후, 즉 현재 step의 최종 weight)
            # 따라서 이 비교는 Step N-1 (optimizer.step() 이후) vs Step N (optimizer.step() 이후)를 비교함
            # 즉, 두 step 사이의 weight 변화를 측정함
            if self.prev_weights is not None and layer_key in self.prev_weights:
                prev_layer_weights = self.prev_weights[layer_key]
                
                # 디버깅: prev_weights가 실제로 이전 step의 값인지 확인
                # (prev_weights는 아직 업데이트되지 않았으므로 이전 step의 값이어야 함)
                if step <= 3:  # 처음 3 step만 디버깅 로그
                    import logging
                    debug_logger = logging.getLogger(__name__)
                    debug_logger.debug(f"  [Step {step}] Comparing prev_weights (from step {step-1}) vs current_weights (from step {step})")
                
                # Load balancer 변화
                if 'load_balancer' in layer_weights and 'load_balancer' in prev_layer_weights:
                    lb_changes = compute_weight_change(
                        prev_layer_weights['load_balancer'],
                        layer_weights['load_balancer']
                    )
                    layer_stats['load_balancer_changes'] = lb_changes
                    
                    # 디버깅: 변화가 있는지 확인
                    if step <= 3:
                        max_lb_change = max([abs(v) for v in lb_changes.values() if isinstance(v, (int, float))], default=0.0)
                        import logging
                        debug_logger = logging.getLogger(__name__)
                        debug_logger.debug(f"    Load balancer max change: {max_lb_change:.2e}")
                
                # Expression projector 변화
                if 'expression_projector' in layer_weights and 'expression_projector' in prev_layer_weights:
                    expr_changes = compute_weight_change(
                        prev_layer_weights['expression_projector'],
                        layer_weights['expression_projector']
                    )
                    layer_stats['expression_projector_changes'] = expr_changes
                    
                    # 디버깅: 변화가 있는지 확인
                    if step <= 3:
                        max_expr_change = max([abs(v) for v in expr_changes.values() if isinstance(v, (int, float))], default=0.0)
                        import logging
                        debug_logger = logging.getLogger(__name__)
                        debug_logger.debug(f"    Expression projector max change: {max_expr_change:.2e}")
            else:
                # prev_weights가 없으면 (첫 step) 변화 계산 불가
                # 이 경우 변화 데이터가 없음을 명시
                if self.prev_weights is None:
                    layer_stats['_note'] = 'No prev_weights - first step, cannot compute changes'
                elif layer_key not in self.prev_weights:
                    layer_stats['_note'] = f'Layer {layer_key} not in prev_weights'
            
            step_stats['layers'][layer_key] = layer_stats
        
        # 히스토리에 추가
        self.weight_history.append(step_stats)
        
        # 메모리 관리: 오래된 히스토리 제거
        if len(self.weight_history) > self.max_history:
            self.weight_history = self.weight_history[-self.max_history:]
        
        # 전체 가중치 저장 (선택적)
        if self.save_full_weights and step % self.save_every_n_steps == 0:
            self.weight_snapshots[step] = current_weights
        
        # 주기적으로 디스크에 저장
        if step % self.save_every_n_steps == 0:
            self._save_to_disk(step)
        
        # CRITICAL: 이전 가중치 업데이트는 비교 후에 수행되어야 함
        # 현재 current_weights는 Step N의 weight이므로, 다음 step (Step N+1)에서 비교할 때 사용됨
        # 즉, Step N+1에서 prev_weights는 Step N의 weight가 됨
        # 이 업데이트는 비교 후에 수행되므로 안전함
        self.prev_weights = current_weights
        
        # 디버깅: prev_weights 업데이트 확인
        if step <= 3:
            import logging
            debug_logger = logging.getLogger(__name__)
            debug_logger.debug(f"  [Step {step}] Updated prev_weights to current_weights (will be used for step {step+1} comparison)")
        
        return step_stats
    
    def _save_to_disk(self, step: int):
        """가중치 히스토리를 디스크에 저장 (atomic write, empty 파일 방지)"""
        try:
            # 디렉토리 존재 확인 및 생성
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            stats_file = self.save_dir / f"router_weight_stats_step_{step}.json"
            # Use unique tmp filename to avoid multi-process collisions
            import uuid, time
            tmp_file = self.save_dir / f"{stats_file.name}.tmp.{os.getpid()}.{time.time_ns()}.{uuid.uuid4().hex}"
            
            # JSON serializable 형태로 변환
            save_data = {
                'step': step,
                'history': self.weight_history[-self.save_every_n_steps:],  # 최근 N step만 저장
            }
            
            # 파일을 tmp에 먼저 저장 후 rename (atomic)
            with open(tmp_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            # Atomic replace to final path
            os.replace(tmp_file, stats_file)
            
            # 파일이 실제로 생성되었는지 확인
            if stats_file.exists():
                file_size = stats_file.stat().st_size
                if file_size <= 0:
                    raise IOError(f"Saved file {stats_file} is empty after write")
            else:
                raise IOError(f"Failed to create file {stats_file}")
            
            # 전체 가중치 스냅샷 저장 (선택적)
            if step in self.weight_snapshots:
                snapshot_file = self.save_dir / f"router_weight_snapshot_step_{step}.pt"
                torch.save(self.weight_snapshots[step], snapshot_file)
                if not snapshot_file.exists():
                    raise IOError(f"Failed to create snapshot file {snapshot_file}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"❌ Failed to save router weights to disk at step {step}: {e}")
            logger.error(f"   Save dir: {self.save_dir}")
            logger.error(f"   Save dir exists: {self.save_dir.exists()}")
            logger.error(f"   Save dir is writable: {os.access(self.save_dir, os.W_OK) if self.save_dir.exists() else False}")
            raise
    
    def get_weight_trajectory(
        self,
        layer_key: str,
        weight_name: str,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
    ) -> List[float]:
        """
        특정 가중치의 step별 변화 trajectory를 반환합니다.
        
        Args:
            layer_key: 레이어 키 (예: 'layer_0')
            weight_name: 가중치 이름 (예: 'load_balancer.weight_ih_mean')
            start_step: 시작 step (None이면 처음부터)
            end_step: 끝 step (None이면 마지막까지)
            
        Returns:
            Step별 가중치 값 리스트
        """
        trajectory = []
        
        for entry in self.weight_history:
            step = entry['step']
            
            if start_step is not None and step < start_step:
                continue
            if end_step is not None and step > end_step:
                break
            
            if layer_key in entry['layers']:
                layer_data = entry['layers'][layer_key]
                
                # weight_name에서 레이어와 가중치 분리
                # 예: 'load_balancer.weight_ih_mean' -> ['load_balancer', 'weight_ih_mean']
                parts = weight_name.split('.', 1)
                if len(parts) == 2:
                    component, stat_name = parts
                    if component in layer_data:
                        if stat_name in layer_data[component]:
                            trajectory.append(layer_data[component][stat_name])
        
        return trajectory
    
    def save_summary(self, output_file: Optional[str] = None):
        """전체 tracking 결과 요약을 저장"""
        if output_file is None:
            output_file = self.save_dir / "router_weight_summary.json"
        else:
            output_file = Path(output_file)
        
        summary = {
            'total_steps': len(self.weight_history),
            'first_step': self.weight_history[0]['step'] if self.weight_history else None,
            'last_step': self.weight_history[-1]['step'] if self.weight_history else None,
            'layers_tracked': list(set(
                layer_key
                for entry in self.weight_history
                for layer_key in entry['layers'].keys()
            )),
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
