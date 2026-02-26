# coding=utf-8
"""
Seqorth MoE의 실제 검증 지표

내부 메커니즘 지표가 아닌, 실제로 중요한 검증 지표들:
1. Expression Semantic Quality: Expression이 진짜 의미있는 표현을 담고 있는지
2. Layer-wise Load Balancing: 모든 layer들이 고르게 사용되는지
3. Information Processing Quality: 정보 흡수/처리가 잘 되고 있는지 (다른 모델과 비교)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import json
from dataclasses import dataclass


@dataclass
class ExpressionAblationResult:
    """Expression projection을 ablation했을 때의 결과"""
    with_expression: float  # Expression 사용 시 성능
    without_expression: float  # Expression 제거 시 성능
    performance_drop: float  # 성능 저하
    relative_drop: float  # 상대적 저하 (%)


class SeqorthSemanticValidator:
    """
    Seqorth MoE의 실제 검증을 위한 지표 계산
    
    핵심 질문:
    1. Expression이 실제로 의미있는 semantic representation을 담고 있는가?
    2. 모든 layer들이 균형있게 활용되는가?
    3. 정보 처리 능력이 다른 모델보다 우수한가?
    """
    
    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.reset()
    
    def reset(self):
        """상태 초기화"""
        self.layer_expert_usage = defaultdict(lambda: defaultdict(int))  # layer -> expert -> count
        self.layer_routing_patterns = defaultdict(list)  # layer -> routing patterns
        self.expression_contributions = []  # Expression의 task별 기여도
        self.information_flow_history = []  # Information processing quality
        
    def analyze_layer_wise_balance(
        self,
        layer_expert_usage_counts: Dict[int, torch.Tensor],  # layer_idx -> [num_experts] usage counts
    ) -> Dict[str, float]:
        """
        Layer-wise load balancing 분석
        
        검증: 모든 layer들이 고르게 사용되는가?
        """
        if not layer_expert_usage_counts:
            return {}
        
        layer_utilizations = []
        layer_entropies = []
        
        for layer_idx, usage_counts in layer_expert_usage_counts.items():
            if usage_counts.numel() == 0:
                continue
            
            # Expert utilization rate (사용된 expert 비율)
            active_experts = (usage_counts > 0).sum().float()
            utilization_rate = active_experts / self.num_experts
            layer_utilizations.append(utilization_rate.item())
            
            # Expert usage entropy (균형도)
            usage_probs = usage_counts.float() / (usage_counts.sum() + 1e-8)
            entropy = -(usage_probs * torch.log(usage_probs + 1e-8)).sum()
            max_entropy = np.log(self.num_experts)
            normalized_entropy = entropy / max_entropy
            layer_entropies.append(normalized_entropy.item())
        
        if not layer_utilizations:
            return {}
        
        # Layer 간 균형도 측정
        utilization_std = np.std(layer_utilizations)  # 낮을수록 균형
        utilization_cv = utilization_std / (np.mean(layer_utilizations) + 1e-8)  # Coefficient of variation
        
        entropy_std = np.std(layer_entropies)
        entropy_mean = np.mean(layer_entropies)
        
        # Early vs Late layers 비교
        early_layers = layer_utilizations[:len(layer_utilizations)//3]
        late_layers = layer_utilizations[-len(layer_utilizations)//3:]
        
        early_late_ratio = np.mean(early_layers) / (np.mean(late_layers) + 1e-8) if late_layers else 1.0
        
        return {
            'layer_utilization_mean': np.mean(layer_utilizations),
            'layer_utilization_std': utilization_std,
            'layer_utilization_cv': utilization_cv,  # 낮을수록 균형 (0.1 이하가 좋음)
            'layer_entropy_mean': entropy_mean,
            'layer_entropy_std': entropy_std,
            'early_late_utilization_ratio': early_late_ratio,  # 1.0에 가까울수록 균형
            'num_layers_analyzed': len(layer_utilizations),
        }
    
    def analyze_expression_semantic_quality(
        self,
        model_with_expression: torch.nn.Module,
        model_without_expression: torch.nn.Module,
        evaluation_fn: Callable[[torch.nn.Module], Dict[str, float]],
        task_name: str = "default",
    ) -> ExpressionAblationResult:
        """
        Expression의 semantic quality 측정
        
        방법: Expression projection을 제거했을 때 성능 저하 측정
        - 성능 저하가 크면 → Expression이 중요한 semantic information을 담고 있음
        - 성능 저하가 작으면 → Expression이 덜 중요함
        
        Args:
            model_with_expression: Expression projection을 사용하는 모델
            model_without_expression: Expression projection을 제거한 모델 (ablation)
            evaluation_fn: 모델을 평가하는 함수 (task별로 다를 수 있음)
            task_name: 평가할 task 이름
        
        Returns:
            ExpressionAblationResult
        """
        # Expression 사용 시 성능
        with_results = evaluation_fn(model_with_expression)
        with_performance = with_results.get('accuracy', with_results.get('score', 0.0))
        
        # Expression 제거 시 성능
        without_results = evaluation_fn(model_without_expression)
        without_performance = without_results.get('accuracy', without_results.get('score', 0.0))
        
        performance_drop = with_performance - without_performance
        relative_drop = (performance_drop / (with_performance + 1e-8)) * 100
        
        result = ExpressionAblationResult(
            with_expression=with_performance,
            without_expression=without_performance,
            performance_drop=performance_drop,
            relative_drop=relative_drop,
        )
        
        self.expression_contributions.append({
            'task': task_name,
            'result': result,
        })
        
        return result
    
    def analyze_information_processing_quality(
        self,
        seqorth_model: torch.nn.Module,
        dense_model: torch.nn.Module,
        baseline_moe_model: Optional[torch.nn.Module] = None,
        evaluation_datasets: Dict[str, Callable] = None,
        information_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        정보 처리 품질 분석
        
        검증: Seqorth MoE가 정보를 더 잘 흡수/처리하는가?
        
        비교 대상:
        1. Dense 모델 (동일 파라미터 수)
        2. Baseline MoE 모델 (Switch Transformer, GShard 등)
        
        측정 지표:
        1. Downstream task performance
        2. Information bottleneck (representation quality)
        3. Training efficiency (같은 step에서의 성능)
        """
        if evaluation_datasets is None:
            evaluation_datasets = {}
        
        results = {
            'vs_dense': {},
            'vs_baseline_moe': {},
            'information_metrics': information_metrics or {},
        }
        
        # Seqorth vs Dense 비교
        for task_name, eval_fn in evaluation_datasets.items():
            seqorth_score = eval_fn(seqorth_model).get('score', 0.0)
            dense_score = eval_fn(dense_model).get('score', 0.0)
            
            improvement = seqorth_score - dense_score
            relative_improvement = (improvement / (dense_score + 1e-8)) * 100
            
            results['vs_dense'][task_name] = {
                'Seqorth': seqorth_score,
                'dense': dense_score,
                'improvement': improvement,
                'relative_improvement': relative_improvement,
            }
        
        # Seqorth vs Baseline MoE 비교
        if baseline_moe_model is not None:
            for task_name, eval_fn in evaluation_datasets.items():
                seqorth_score = eval_fn(seqorth_model).get('score', 0.0)
                baseline_score = eval_fn(baseline_moe_model).get('score', 0.0)
                
                improvement = seqorth_score - baseline_score
                relative_improvement = (improvement / (baseline_score + 1e-8)) * 100
                
                results['vs_baseline_moe'][task_name] = {
                    'Seqorth': seqorth_score,
                    'baseline': baseline_score,
                    'improvement': improvement,
                    'relative_improvement': relative_improvement,
                }
        
        self.information_flow_history.append(results)
        return results
    
    def compute_representation_quality(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        probe_datasets: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Representation quality 측정
        
        방법: Linear probing을 통한 representation quality 측정
        - 높은 quality → 정보가 잘 보존됨
        - 낮은 quality → 정보 손실
        
        Args:
            model: 평가할 모델
            tokenizer: 토크나이저
            probe_datasets: Linear probing용 데이터셋 리스트
                           각 항목은 {'name': str, 'data': List, 'labels': List}
        
        Returns:
            각 데이터셋에 대한 representation quality 점수
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        results = {}
        
        model.eval()
        with torch.no_grad():
            for dataset in probe_datasets:
                dataset_name = dataset['name']
                texts = dataset['data']
                labels = dataset['labels']
                
                # Extract representations
                representations = []
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs, output_hidden_states=True)
                    # Use last layer hidden states (mean pooling)
                    hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]
                    representation = hidden_states.mean(dim=1).cpu().numpy()  # [batch, hidden]
                    representations.append(representation[0])
                
                representations = np.array(representations)
                
                # Linear probing
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(representations, labels)
                
                predictions = clf.predict(representations)
                accuracy = accuracy_score(labels, predictions)
                
                results[dataset_name] = accuracy
        
        return results
    
    def analyze_expert_specialization_functional(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        task_samples: List[Dict[str, Any]],  # [{'input': str, 'output': str, 'task_type': str}]
    ) -> Dict[str, Any]:
        """
        Expert specialization의 실제 기능적 검증
        
        방법: 각 expert가 특정 task type에 특화되어 있는지 확인
        - Task-specific routing: 특정 task에서 특정 expert들이 더 활성화되는가?
        - Expert-task correlation: Expert와 task type 간의 상관관계
        
        Returns:
            Expert별 task specialization score
        """
        model.eval()
        expert_task_counts = defaultdict(lambda: defaultdict(int))  # expert -> task -> count
        
        with torch.no_grad():
            for sample in task_samples:
                task_type = sample.get('task_type', 'unknown')
                input_text = sample['input']
                
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Forward pass와 expert selection 추출
                # (실제 구현은 모델 구조에 따라 다름)
                outputs = model(**inputs, output_hidden_states=True)
                
                # 각 layer에서 선택된 expert 추출
                # 이는 모델의 forward hook을 통해 추출해야 함
                # 여기서는 예시만 제공
                
                # expert_task_counts[expert_idx][task_type] += 1
        
        # Expert-task specialization score 계산
        specialization_scores = {}
        for expert_idx in range(self.num_experts):
            task_counts = expert_task_counts[expert_idx]
            if not task_counts:
                continue
            
            total_count = sum(task_counts.values())
            # Entropy: 낮을수록 특정 task에 특화됨
            task_probs = [count / total_count for count in task_counts.values()]
            entropy = -sum(p * np.log(p + 1e-8) for p in task_probs)
            max_entropy = np.log(len(task_probs)) if task_probs else 0
            specialization_score = 1.0 - (entropy / (max_entropy + 1e-8))  # 높을수록 특화
            
            specialization_scores[expert_idx] = {
                'specialization_score': specialization_score,
                'dominant_task': max(task_counts.items(), key=lambda x: x[1])[0] if task_counts else None,
                'task_distribution': dict(task_counts),
            }
        
        return specialization_scores
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """전체 학습 과정에 대한 집계 지표"""
        aggregated = {
            'expression_contributions': [],
            'information_processing_comparisons': self.information_flow_history,
        }
        
        # Expression contribution 요약
        if self.expression_contributions:
            for contrib in self.expression_contributions:
                aggregated['expression_contributions'].append({
                    'task': contrib['task'],
                    'performance_drop': contrib['result'].performance_drop,
                    'relative_drop': contrib['result'].relative_drop,
                })
        
        return aggregated
    
    def save_analysis(self, filepath: str):
        """분석 결과 저장"""
        analysis_data = {
            'aggregated_metrics': self.get_aggregated_metrics(),
            'layer_expert_usage': {
                str(k): {str(ek): int(ev) for ek, ev in v.items()}
                for k, v in self.layer_expert_usage.items()
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)


def create_ablation_model(model: torch.nn.Module, remove_expression: bool = True) -> torch.nn.Module:
    """
    Expression projection을 제거한 ablation 모델 생성
    
    Args:
        model: 원본 모델
        remove_expression: True면 expression projection 제거
    
    Returns:
        Ablation 모델 (in-place modification이므로 원본도 변경됨)
    """
    import copy
    ablated_model = copy.deepcopy(model)
    
    if remove_expression:
        for module in ablated_model.modules():
            if hasattr(module, 'router') and hasattr(module.router, 'expression_projector'):
                # Expression projector를 identity로 대체
                # 또는 routing에서 expression을 사용하지 않도록 수정
                # (구체적 구현은 모델 구조에 따라 다름)
                pass
    
    return ablated_model

