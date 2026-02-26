# coding=utf-8
"""
Seqorth MoE 논문 검증을 위한 분석 도구

이 모듈은 Seqorth MoE의 핵심 주장을 검증하기 위한 다양한 지표를 제공합니다:
1. Expert Specialization Metrics: Expert들이 실제로 다른 기능을 수행하는지
2. Gram Matrix Quality: Gram matrix의 직교성 수준
3. Routing Decision Quality: 라우팅 결정의 품질
4. Sequential Context Utilization: GRU 기반 sequential routing의 효과
5. Computational Efficiency: 계산 오버헤드 분석
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json


class SeqorthAnalyzer:
    """
    Seqorth MoE의 핵심 주장을 검증하기 위한 분석 도구
    
    논문에서 검증해야 할 주요 포인트:
    1. Gram matrix 기반 orthogonal constraint가 expert diversity를 향상시키는가?
    2. Orthogonal expression projection이 expert specialization을 촉진하는가?
    3. Sequential routing (GRU)이 context-aware routing을 가능하게 하는가?
    4. Domain scoring (cosine similarity + speciality penalty)이 효과적인가?
    """
    
    def __init__(self, num_experts: int, router_dim: int = 128, capacity_factor: float = 1.25):
        self.num_experts = num_experts
        self.router_dim = router_dim
        self.capacity_factor = capacity_factor  # Capacity factor for capacity usage calculation
        self.reset()
    
    def reset(self):
        """분석 결과 초기화"""
        self.expert_activation_patterns = defaultdict(list)
        self.gram_matrix_history = []
        self.expression_projection_history = []
        self.routing_decision_history = []
        self.cosine_similarity_history = []
        self.speciality_penalty_history = []
        
        # Load balancing metrics
        self.expert_token_counts_history = []  # List of [num_experts] tensors
        self.expert_activation_counts_history = []  # List of [num_experts] tensors
        self.load_balancing_coefficient_history = []
        self.expert_utilization_history = []
        
        # Additional metrics from recent papers
        self.maxvio_history = []  # Maximum violation (Loss-free balancing)
        self.aux_loss_history = []  # Auxiliary loss (Switch Transformer, DeepSpeed MoE)
        self.lpr_history = []  # Layer-wise Performance Ratio
        self.deepspeed_metrics_history = []  # DeepSpeed MoE metrics
        
        # Specialization metrics history
        self.expert_utilization_rate_history = []
        self.expert_diversity_score_history = []
        self.expert_similarity_mean_history = []
        self.expert_specialization_strength_history = []
        
        # 논문 필수 지표 히스토리
        self.expert_overlap_history = []  # Expert Overlap (Jaccard Similarity)
        self.gini_coefficient_history = []  # Gini Coefficient
        self.min_max_expert_load_ratio_history = []  # Min-max Expert Load Ratio
        
    def analyze_routing_step(
        self,
        routing_logits: torch.Tensor,  # [batch*seq, num_experts, router_dim]
        expression_logits: torch.Tensor,  # [batch*seq, num_experts, router_dim]
        selected_experts: torch.Tensor,  # [batch*seq, top_k]
        routing_weights: torch.Tensor,  # [batch*seq, top_k]
        speciality_penalty: float,
        cosine_similarities: torch.Tensor,  # [batch, seq, num_experts]
        expression_loss: float,
        expert_outputs: Optional[Dict[int, torch.Tensor]] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        단일 routing step 분석
        
        Returns:
            분석 결과 딕셔너리
        """
        metrics = {}
        
        # 1. Gram Matrix Quality Metrics
        gram_metrics = self._compute_gram_matrix_quality(routing_logits)
        metrics.update(gram_metrics)
        
        # 2. Expert Specialization Metrics
        specialization_metrics = self._compute_expert_specialization(
            routing_logits, expression_logits, selected_experts, routing_weights
        )
        metrics.update(specialization_metrics)
        
        # 3. Routing Decision Quality
        routing_quality = self._compute_routing_decision_quality(
            selected_experts, routing_weights, cosine_similarities
        )
        metrics.update(routing_quality)
        
        # 4. Expression Projection Effectiveness
        expression_metrics = self._compute_expression_projection_quality(
            expression_logits, routing_logits, cosine_similarities
        )
        metrics.update(expression_metrics)
        
        # 5. Speciality Penalty Contribution
        metrics['speciality_penalty'] = float(speciality_penalty)
        metrics['expression_loss'] = float(expression_loss)
        
        # 6. Expert Activation Diversity (if expert outputs available)
        if expert_outputs is not None:
            diversity_metrics = self._compute_expert_output_diversity(expert_outputs)
            metrics.update(diversity_metrics)
        
        # 10. Inference Cost / FLOPs / Utilized Parameters (추가)
        # Note: FLOPs 계산은 별도로 수행해야 하므로 여기서는 활성화된 expert 수만 계산
        inference_cost_metrics = self._compute_inference_cost_metrics(
            selected_experts, routing_weights
        )
        metrics.update(inference_cost_metrics)
        
        # 7. Context Utilization (if hidden states available)
        if hidden_states is not None:
            context_metrics = self._compute_context_utilization(hidden_states, selected_experts)
            metrics.update(context_metrics)
        
        # 9. Routing Consistency / Locality Metrics (추가)
        consistency_metrics = self._compute_routing_consistency_metrics(
            selected_experts, routing_weights
        )
        metrics.update(consistency_metrics)
        
        # 8. Load Balancing Metrics (필수 - 논문에 포함)
        # Capacity factor는 일반적으로 1.0-2.0 사이 (기본값 1.25)
        capacity_factor = getattr(self, 'capacity_factor', 1.25)
        load_balancing_metrics = self._compute_load_balancing_metrics(
            selected_experts, routing_weights, capacity_factor=capacity_factor
        )
        metrics.update(load_balancing_metrics)
        
        # 히스토리 저장
        self.gram_matrix_history.append(gram_metrics.get('gram_matrix_orthogonality', 0.0))
        self.speciality_penalty_history.append(speciality_penalty)
        self.cosine_similarity_history.append(cosine_similarities.mean().item())
        
        # Load balancing 히스토리 저장
        if 'expert_token_counts' in load_balancing_metrics:
            self.expert_token_counts_history.append(load_balancing_metrics['expert_token_counts'])
        
        # 논문 필수 지표 히스토리 저장
        if 'gini_coefficient' in load_balancing_metrics:
            if not hasattr(self, 'gini_coefficient_history'):
                self.gini_coefficient_history = []
            self.gini_coefficient_history.append(load_balancing_metrics['gini_coefficient'])
        
        if 'min_max_expert_load_ratio' in load_balancing_metrics:
            if not hasattr(self, 'min_max_expert_load_ratio_history'):
                self.min_max_expert_load_ratio_history = []
            self.min_max_expert_load_ratio_history.append(load_balancing_metrics['min_max_expert_load_ratio'])
        
        if 'expert_overlap_mean' in specialization_metrics:
            if not hasattr(self, 'expert_overlap_history'):
                self.expert_overlap_history = []
            self.expert_overlap_history.append(specialization_metrics['expert_overlap_mean'])
        if 'load_balancing_coefficient' in load_balancing_metrics:
            self.load_balancing_coefficient_history.append(load_balancing_metrics['load_balancing_coefficient'])
        if 'expert_utilization_rate' in load_balancing_metrics:
            self.expert_utilization_rate_history.append(load_balancing_metrics['expert_utilization_rate'])
        if 'maxvio' in load_balancing_metrics:
            self.maxvio_history.append(load_balancing_metrics['maxvio'])
        if 'aux_loss' in load_balancing_metrics:
            self.aux_loss_history.append(load_balancing_metrics['aux_loss'])
        if 'lpr' in load_balancing_metrics:
            self.lpr_history.append(load_balancing_metrics['lpr'])
        if 'expert_efficiency' in load_balancing_metrics:
            self.deepspeed_metrics_history.append({
                'expert_efficiency': load_balancing_metrics['expert_efficiency'],
                'expert_capacity_utilization': load_balancing_metrics.get('expert_capacity_utilization', 0.0),
                'load_variance': load_balancing_metrics.get('load_variance', 0.0),
            })
        
        # Specialization 히스토리 저장
        if 'expert_utilization_rate' in specialization_metrics:
            # 이미 load_balancing에서 저장했으므로 중복 방지
            pass
        if 'expert_diversity_score' in specialization_metrics:
            self.expert_diversity_score_history.append(specialization_metrics['expert_diversity_score'])
        if 'expert_similarity_mean' in specialization_metrics:
            self.expert_similarity_mean_history.append(specialization_metrics['expert_similarity_mean'])
        if 'expert_specialization_strength' in specialization_metrics:
            self.expert_specialization_strength_history.append(specialization_metrics['expert_specialization_strength'])
        
        return metrics
    
    def _compute_gram_matrix_quality(self, routing_logits: torch.Tensor) -> Dict[str, float]:
        """
        Gram matrix의 품질 지표 계산
        
        검증 포인트: Gram matrix가 identity matrix에 가까워질수록 expert들이 orthogonal한가?
        """
        # routing_logits: [batch*seq, num_experts, router_dim]
        # 또는 [batch, seq, num_experts, router_dim]
        
        if routing_logits.dim() == 3:
            # [batch*seq, num_experts, router_dim] -> [num_experts, router_dim] (평균)
            routing_mean = routing_logits.mean(dim=0)  # [num_experts, router_dim]
        elif routing_logits.dim() == 4:
            # [batch, seq, num_experts, router_dim] -> [num_experts, router_dim] (평균)
            routing_mean = routing_logits.mean(dim=[0, 1])  # [num_experts, router_dim]
        else:
            raise ValueError(f"Unexpected routing_logits shape: {routing_logits.shape}")
        
        # Normalize
        routing_normalized = F.normalize(routing_mean, dim=-1)  # [num_experts, router_dim]
        
        # Compute Gram matrix: G = R @ R^T
        gram_matrix = torch.matmul(routing_normalized, routing_normalized.T)  # [num_experts, num_experts]
        
        # Identity matrix
        identity = torch.eye(self.num_experts, device=gram_matrix.device, dtype=gram_matrix.dtype)
        
        # Orthogonality metrics
        gram_deviation = gram_matrix - identity
        gram_orthogonality = 1.0 - torch.norm(gram_deviation, p='fro') / (self.num_experts * np.sqrt(2))
        
        # Diagonal dominance: diagonal elements should be close to 1
        diagonal_quality = gram_matrix.diag().mean().item()
        
        # Off-diagonal sparsity: off-diagonal elements should be close to 0
        off_diagonal_mask = ~torch.eye(self.num_experts, dtype=bool, device=gram_matrix.device)
        off_diagonal_mean = gram_matrix[off_diagonal_mask].abs().mean().item()
        
        # Condition number (for numerical stability)
        try:
            eigenvals = torch.linalg.eigvalsh(gram_matrix)
            condition_number = (eigenvals.max() / eigenvals.min()).item() if eigenvals.min() > 1e-8 else float('inf')
        except:
            condition_number = float('inf')
        
        return {
            'gram_matrix_orthogonality': gram_orthogonality.item(),
            'gram_diagonal_quality': diagonal_quality,
            'gram_off_diagonal_sparsity': off_diagonal_mean,
            'gram_condition_number': condition_number,
            'gram_frobenius_norm': torch.norm(gram_matrix, p='fro').item(),
        }
    
    def _compute_expert_specialization(
        self,
        routing_logits: torch.Tensor,
        expression_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Expert specialization 지표 계산 (논문 필수 지표)
        
        검증 포인트: 각 expert가 서로 다른 input pattern에 반응하는가?
        
        논문에서 사용하는 specialization 지표:
        1. Expert similarity matrix (낮을수록 specialized)
        2. Expert activation entropy (균형도 측정)
        3. Expression-routing alignment (specialization quality)
        4. Expert diversity score
        """
        # Expert별 activation pattern 분석
        # selected_experts: [batch*seq, top_k]
        # routing_weights: [batch*seq, top_k]
        
        selected_flat = selected_experts.flatten()  # [batch*seq * top_k]
        weights_flat = routing_weights.flatten()  # [batch*seq * top_k]
        
        # Expert별 weighted activation count
        expert_activation_counts = torch.zeros(self.num_experts, device=selected_experts.device)
        expert_activation_weights = torch.zeros(self.num_experts, device=selected_experts.device)
        
        for expert_idx in range(self.num_experts):
            mask = (selected_flat == expert_idx)
            expert_activation_counts[expert_idx] = mask.sum().float()
            expert_activation_weights[expert_idx] = weights_flat[mask].sum()
        
        # Activation distribution entropy (higher = more balanced)
        activation_probs = expert_activation_weights / (expert_activation_weights.sum() + 1e-8)
        activation_entropy = -(activation_probs * torch.log(activation_probs + 1e-8)).sum()
        max_entropy = np.log(self.num_experts)
        normalized_entropy = activation_entropy / max_entropy
        
        # Expert specialization score: 각 expert가 고유한 표현 공간을 사용하는가?
        # Expression projection과 routing logits 간의 cosine similarity 분석
        if routing_logits.dim() == 3:
            routing_mean = routing_logits.mean(dim=0)  # [num_experts, router_dim]
            expression_mean = expression_logits.mean(dim=0)  # [num_experts, router_dim]
        elif routing_logits.dim() == 4:
            routing_mean = routing_logits.mean(dim=[0, 1])  # [num_experts, router_dim]
            expression_mean = expression_logits.mean(dim=[0, 1])  # [num_experts, router_dim]
        else:
            routing_mean = routing_logits
            expression_mean = expression_logits
        
        # Expert 간 cosine similarity (낮을수록 더 specialized)
        routing_normalized = F.normalize(routing_mean, dim=-1)
        expert_similarity_matrix = torch.matmul(routing_normalized, routing_normalized.T)
        off_diagonal_mask = ~torch.eye(self.num_experts, dtype=bool, device=expert_similarity_matrix.device)
        expert_similarity_mean = expert_similarity_matrix[off_diagonal_mask].mean()
        expert_similarity_std = expert_similarity_matrix[off_diagonal_mask].std()
        
        # Expert diversity score: 1 - mean_similarity (높을수록 더 diverse/specialized)
        expert_diversity_score = 1.0 - expert_similarity_mean
        
        # Expression-routing alignment: 각 expert의 expression과 routing이 얼마나 일치하는가?
        expression_normalized = F.normalize(expression_mean, dim=-1)
        alignment_scores = F.cosine_similarity(expression_normalized, routing_normalized, dim=-1)
        alignment_mean = alignment_scores.mean()
        alignment_std = alignment_scores.std()
        
        # Expert specialization strength: 각 expert가 얼마나 명확하게 구분되는가?
        # Similarity matrix의 off-diagonal 요소들의 분산 (높을수록 expert들이 더 구분됨)
        specialization_strength = expert_similarity_matrix[off_diagonal_mask].var().item()
        
        # Expert concentration: 특정 expert에 집중되는 정도 (낮을수록 균형)
        max_activation_prob = activation_probs.max().item()
        expert_concentration = max_activation_prob * self.num_experts  # Normalized
        
        # Expert Overlap (Jaccard Similarity) - 논문 필수 지표
        # 각 expert pair 간 token set의 Jaccard similarity 계산
        # selected_experts: [batch*seq, top_k]
        batch_seq_len = selected_experts.size(0)
        expert_token_sets = {}
        for expert_idx in range(self.num_experts):
            # 각 expert가 처리한 토큰 인덱스 집합
            expert_mask = (selected_experts == expert_idx).any(dim=1)  # [batch*seq]
            expert_token_sets[expert_idx] = set(torch.where(expert_mask)[0].cpu().numpy().tolist())
        
        # 모든 expert pair 간 Jaccard similarity 계산
        expert_overlaps = []
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                set_i = expert_token_sets.get(i, set())
                set_j = expert_token_sets.get(j, set())
                
                if len(set_i) == 0 and len(set_j) == 0:
                    # 둘 다 사용되지 않음
                    jaccard = 1.0  # 완전히 겹침 (둘 다 비어있음)
                elif len(set_i) == 0 or len(set_j) == 0:
                    # 하나만 사용됨
                    jaccard = 0.0  # 겹침 없음
                else:
                    # Jaccard similarity: |A ∩ B| / |A ∪ B|
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    jaccard = intersection / union if union > 0 else 0.0
                
                expert_overlaps.append(jaccard)
        
        # 평균 Expert Overlap (낮을수록 specialization 우수)
        expert_overlap_mean = np.mean(expert_overlaps) if expert_overlaps else 0.0
        expert_overlap_std = np.std(expert_overlaps) if expert_overlaps else 0.0
        expert_overlap_max = np.max(expert_overlaps) if expert_overlaps else 0.0
        
        return {
            'expert_activation_entropy': normalized_entropy.item(),
            'expert_similarity_mean': expert_similarity_mean.item(),
            'expert_similarity_std': expert_similarity_std.item(),
            'expert_diversity_score': expert_diversity_score.item(),
            'expert_routing_expression_alignment': alignment_mean.item(),
            'expert_routing_expression_alignment_std': alignment_std.item(),
            'expert_specialization_strength': specialization_strength,
            'expert_concentration': expert_concentration,
            'expert_utilization_rate': (expert_activation_counts > 0).float().mean().item(),
            # Expert Overlap (Jaccard Similarity) - 논문 필수 지표
            'expert_overlap_mean': expert_overlap_mean,
            'expert_overlap_std': expert_overlap_std,
            'expert_overlap_max': expert_overlap_max,
        }
    
    def _compute_routing_decision_quality(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        cosine_similarities: torch.Tensor,
    ) -> Dict[str, float]:
        """
        라우팅 결정의 품질 지표 계산
        
        검증 포인트: Domain scoring (cosine similarity + speciality penalty)이 효과적인가?
        """
        # Routing confidence: 선택된 expert들의 가중치 분포
        routing_entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1).mean()
        max_routing_entropy = np.log(routing_weights.size(-1))
        normalized_routing_entropy = routing_entropy / max_routing_entropy
        
        # Cosine similarity utilization: domain scoring이 cosine similarity를 얼마나 활용하는가?
        # cosine_similarities: [batch, seq, num_experts]
        cosine_mean = cosine_similarities.mean().item()
        cosine_std = cosine_similarities.std().item()
        
        # Top-k selection quality: 선택된 expert들의 cosine similarity가 높은가?
        selected_experts_flat = selected_experts.flatten()  # [batch*seq * top_k]
        batch_size, seq_len = cosine_similarities.shape[:2]
        cosine_flat = cosine_similarities.view(batch_size * seq_len, -1)  # [batch*seq, num_experts]
        
        # 각 선택된 expert의 cosine similarity
        selected_cosine_scores = []
        for i in range(selected_experts.size(0)):
            for j in range(selected_experts.size(1)):
                expert_idx = selected_experts[i, j].item()
                cosine_idx = i % (batch_size * seq_len)
                if cosine_idx < cosine_flat.size(0):
                    selected_cosine_scores.append(cosine_flat[cosine_idx, expert_idx].item())
        
        selected_cosine_mean = np.mean(selected_cosine_scores) if selected_cosine_scores else 0.0
        
        return {
            'routing_confidence': normalized_routing_entropy.item(),
            'cosine_similarity_mean': cosine_mean,
            'cosine_similarity_std': cosine_std,
            'selected_expert_cosine_mean': selected_cosine_mean,
        }
    
    def _compute_expression_projection_quality(
        self,
        expression_logits: torch.Tensor,
        routing_logits: torch.Tensor,
        cosine_similarities: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Expression projection의 효과성 지표 계산
        
        검증 포인트: Orthogonal expression projection이 expert specialization을 촉진하는가?
        """
        # Expression projection의 orthogonal property
        if expression_logits.dim() == 3:
            expression_mean = expression_logits.mean(dim=0)  # [num_experts, router_dim]
        elif expression_logits.dim() == 4:
            expression_mean = expression_logits.mean(dim=[0, 1])  # [num_experts, router_dim]
        else:
            expression_mean = expression_logits
        
        expression_normalized = F.normalize(expression_mean, dim=-1)
        expression_gram = torch.matmul(expression_normalized, expression_normalized.T)
        expression_identity = torch.eye(self.num_experts, device=expression_gram.device, dtype=expression_gram.dtype)
        expression_orthogonality = 1.0 - torch.norm(expression_gram - expression_identity, p='fro') / (self.num_experts * np.sqrt(2))
        
        # Expression-routing complementarity: 서로 다른 정보를 담고 있는가?
        if routing_logits.dim() == 3:
            routing_mean = routing_logits.mean(dim=0)
        elif routing_logits.dim() == 4:
            routing_mean = routing_logits.mean(dim=[0, 1])
        else:
            routing_mean = routing_logits
        
        routing_normalized = F.normalize(routing_mean, dim=-1)
        expression_routing_similarity = F.cosine_similarity(expression_normalized, routing_normalized, dim=-1).mean()
        
        # Complementarity score: 낮을수록 expression과 routing이 서로 다른 정보를 담고 있음
        complementarity = 1.0 - expression_routing_similarity
        
        return {
            'expression_projection_orthogonality': expression_orthogonality.item(),
            'expression_routing_complementarity': complementarity.item(),
        }
    
    def _compute_expert_output_diversity(self, expert_outputs: Dict[int, torch.Tensor]) -> Dict[str, float]:
        """
        Expert output의 다양성 지표 계산
        
        검증 포인트: 각 expert가 실제로 다른 출력을 생성하는가?
        """
        if not expert_outputs:
            return {}
        
        expert_outputs_list = []
        for expert_idx in sorted(expert_outputs.keys()):
            output = expert_outputs[expert_idx]  # [num_tokens, hidden_dim]
            expert_outputs_list.append(output.mean(dim=0))  # [hidden_dim]
        
        if len(expert_outputs_list) < 2:
            return {}
        
        expert_outputs_tensor = torch.stack(expert_outputs_list)  # [num_experts, hidden_dim]
        expert_outputs_normalized = F.normalize(expert_outputs_tensor, dim=-1)
        
        # Expert output similarity matrix
        output_similarity = torch.matmul(expert_outputs_normalized, expert_outputs_normalized.T)
        off_diagonal_mask = ~torch.eye(len(expert_outputs_list), dtype=bool, device=output_similarity.device)
        output_diversity = 1.0 - output_similarity[off_diagonal_mask].mean().item()
        
        return {
            'expert_output_diversity': output_diversity,
        }
    
    def _compute_context_utilization(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Sequential context 활용도 지표 계산
        
        검증 포인트: GRU 기반 sequential routing이 context를 활용하는가?
        """
        # Sequential patterns: 연속된 토큰들이 같은 expert를 선택하는 경향
        # selected_experts: [batch*seq, top_k]
        batch_seq_len = selected_experts.size(0)
        
        if batch_seq_len < 2:
            return {}
        
        # 첫 번째 expert 선택 패턴
        first_expert = selected_experts[:, 0]  # [batch*seq]
        
        # Sequential consistency: 연속된 토큰들이 같은 expert를 선택하는 비율
        sequential_changes = (first_expert[1:] != first_expert[:-1]).sum().float()
        sequential_consistency = 1.0 - (sequential_changes / (batch_seq_len - 1))
        
        return {
            'sequential_routing_consistency': sequential_consistency.item(),
        }
    
    def _compute_routing_consistency_metrics(
        self,
        selected_experts: torch.Tensor,  # [batch*seq, top_k]
        routing_weights: torch.Tensor,  # [batch*seq, top_k]
    ) -> Dict[str, float]:
        """
        Routing Consistency / Locality Metrics (논문 필수 지표)
        
        Measures:
        1. Segment Routing Performance (SRP): 연속된 토큰들이 같은 expert를 선택하는 비율
        2. Segment Cache Hit Rate (SCH): 캐시 효율을 위한 동일 expert 재사용 비율
        3. Routing locality: 인접한 토큰들이 비슷한 expert를 선택하는 경향
        """
        batch_seq_len = selected_experts.size(0)
        top_k = selected_experts.size(1)
        
        if batch_seq_len < 2:
            return {}
        
        # Top-1 expert 선택 패턴
        top1_expert = selected_experts[:, 0]  # [batch*seq]
        
        # 1. Sequential Routing Consistency (SRP-like)
        # 연속된 토큰들이 같은 top-1 expert를 선택하는 비율
        sequential_matches = (top1_expert[1:] == top1_expert[:-1]).sum().float()
        sequential_routing_consistency = sequential_matches / (batch_seq_len - 1)
        
        # 2. Top-k overlap: 연속된 토큰들의 top-k expert 집합이 겹치는 비율
        top_k_overlaps = []
        for i in range(batch_seq_len - 1):
            experts_i = set(selected_experts[i].cpu().numpy())
            experts_next = set(selected_experts[i + 1].cpu().numpy())
            overlap = len(experts_i & experts_next) / len(experts_i | experts_next) if (experts_i | experts_next) else 0.0
            top_k_overlaps.append(overlap)
        avg_top_k_overlap = np.mean(top_k_overlaps) if top_k_overlaps else 0.0
        
        # 3. Routing locality: 인접 토큰들이 비슷한 routing weight를 가지는지
        # 첫 번째 expert의 weight를 비교
        top1_weights = routing_weights[:, 0]  # [batch*seq]
        weight_correlation = 0.0
        if batch_seq_len > 1:
            # 인접 토큰 간 weight 차이
            weight_diffs = (top1_weights[1:] - top1_weights[:-1]).abs()
            weight_locality = 1.0 - weight_diffs.mean().item()  # 낮은 차이 = 높은 locality
        
        # 4. Expert reuse rate: 같은 expert가 연속으로 선택되는 비율
        expert_reuse_rate = sequential_routing_consistency.item()
        
        return {
            'sequential_routing_consistency': sequential_routing_consistency.item(),
            'top_k_overlap': avg_top_k_overlap,
            'routing_locality': weight_locality if batch_seq_len > 1 else 0.0,
            'expert_reuse_rate': expert_reuse_rate,
        }
    
    def _compute_load_balancing_metrics(
        self,
        selected_experts: torch.Tensor,  # [batch*seq, top_k]
        routing_weights: torch.Tensor,  # [batch*seq, top_k]
        capacity_factor: Optional[float] = None,  # Capacity factor (c in c * T/E)
    ) -> Dict[str, Any]:
        """
        Load balancing 지표 계산 (논문 필수 지표)
        
        MoE 논문에서 표준적으로 사용하는 지표들:
        1. Expert별 처리 토큰 수 (per-expert token count) ✅
        2. Load balancing coefficient (CV - coefficient of variation) ✅
        3. Expert utilization rate ✅
        4. Load imbalance ratio ✅
        5. Capacity factor / capacity usage ✅ (추가)
        6. Routing sparsity (experts per token) ✅ (추가)
        7. Fraction of active experts / sparsity ratio ✅ (추가)
        
        Returns:
            Load balancing 관련 지표 딕셔너리
        """
        # selected_experts: [batch*seq, top_k]
        # routing_weights: [batch*seq, top_k]
        
        batch_seq_len = selected_experts.size(0)
        top_k = selected_experts.size(1)
        total_tokens = batch_seq_len
        
        # Expert별 토큰 수 계산 (각 토큰이 top_k개 expert에 할당되므로 가중치 고려)
        expert_token_counts = torch.zeros(self.num_experts, device=selected_experts.device, dtype=torch.float32)
        expert_weighted_counts = torch.zeros(self.num_experts, device=selected_experts.device, dtype=torch.float32)
        
        # 각 expert가 처리한 토큰 수 (가중치 포함)
        for i in range(batch_seq_len):
            for j in range(top_k):
                expert_idx = selected_experts[i, j].item()
                weight = routing_weights[i, j].item()
                expert_token_counts[expert_idx] += 1.0
                expert_weighted_counts[expert_idx] += weight
        
        # Expert별 활성화 횟수 (가중치 없이)
        expert_activation_counts = torch.zeros(self.num_experts, device=selected_experts.device, dtype=torch.long)
        selected_flat = selected_experts.flatten()
        for expert_idx in range(self.num_experts):
            expert_activation_counts[expert_idx] = (selected_flat == expert_idx).sum().long()
        
        # Load Balancing Coefficient (CV - Coefficient of Variation)
        # Switch Transformer, GShard 등에서 사용하는 표준 지표
        # CV = std / mean (낮을수록 균형이 좋음)
        token_counts_mean = expert_token_counts.mean().item()
        token_counts_std = expert_token_counts.std().item()
        load_balancing_coefficient = token_counts_std / (token_counts_mean + 1e-8)
        
        # Weighted load balancing coefficient
        weighted_counts_mean = expert_weighted_counts.mean().item()
        weighted_counts_std = expert_weighted_counts.std().item()
        weighted_load_balancing_coefficient = weighted_counts_std / (weighted_counts_mean + 1e-8)
        
        # Load Imbalance Ratio: max / mean (1에 가까울수록 균형이 좋음)
        load_imbalance_ratio = expert_token_counts.max().item() / (token_counts_mean + 1e-8)
        weighted_load_imbalance_ratio = expert_weighted_counts.max().item() / (weighted_counts_mean + 1e-8)
        
        # Expert Utilization Rate: 실제로 사용된 expert 비율
        expert_utilization_rate = (expert_token_counts > 0).float().mean().item()
        
        # Expert별 처리 비율 (정규화)
        total_tokens = expert_token_counts.sum().item()
        expert_token_proportions = (expert_token_counts / (total_tokens + 1e-8)).cpu().numpy().tolist()
        
        # Ideal distribution: 모든 expert가 균등하게 처리한다면 1/num_experts
        ideal_proportion = 1.0 / self.num_experts
        expert_proportion_entropy = -(torch.tensor(expert_token_proportions) * 
                                     torch.log(torch.tensor(expert_token_proportions) + 1e-8)).sum().item()
        max_entropy = np.log(self.num_experts)
        normalized_proportion_entropy = expert_proportion_entropy / max_entropy
        
        # Expert별 평균 routing weight
        expert_avg_weights = torch.zeros(self.num_experts, device=selected_experts.device, dtype=torch.float32)
        expert_weight_counts = torch.zeros(self.num_experts, device=selected_experts.device, dtype=torch.float32)
        
        for i in range(batch_seq_len):
            for j in range(top_k):
                expert_idx = selected_experts[i, j].item()
                weight = routing_weights[i, j].item()
                expert_avg_weights[expert_idx] += weight
                expert_weight_counts[expert_idx] += 1.0
        
        expert_avg_weights = expert_avg_weights / (expert_weight_counts + 1e-8)
        expert_avg_weight_mean = expert_avg_weights.mean().item()
        expert_avg_weight_std = expert_avg_weights.std().item()
        
        # MaxVio (Maximum Violation) - Loss-free balancing paper
        # Measures the maximum deviation of expert load from the mean load
        mean_load = token_counts_mean
        maxvio = (expert_token_counts - mean_load).abs().max().item()
        normalized_maxvio = maxvio / (mean_load + 1e-8)
        
        # Aux Loss (Auxiliary Loss) - Switch Transformer, DeepSpeed MoE
        # Computes the auxiliary loss for load balancing
        # Formula: num_experts * sum(f_i * P_i) where f_i is fraction of tokens, P_i is average routing probability
        # For simplicity, we use expert_token_counts as proxy for routing probabilities
        expert_fractions = expert_token_counts / (total_tokens + 1e-8)
        # Average routing weight per expert (proxy for routing probability)
        expert_routing_probs = expert_avg_weights / (expert_avg_weights.sum() + 1e-8)
        aux_loss = self.num_experts * (expert_fractions * expert_routing_probs).sum().item()
        
        # LPR (Layer-wise Performance Ratio) - simplified version
        # Measures the ratio of expert performance contribution
        # For now, we use routing weight variance as a proxy
        lpr = expert_avg_weight_std / (expert_avg_weight_mean + 1e-8)
        
        # DeepSpeed MoE metrics
        # 1. Expert capacity utilization (how well experts are utilized)
        expert_capacity_utilization = expert_utilization_rate
        
        # 2. Load variance (normalized)
        load_variance = token_counts_std ** 2 / ((mean_load + 1e-8) ** 2)
        
        # 3. Expert efficiency (inverse of imbalance)
        expert_efficiency = 1.0 / (load_imbalance_ratio + 1e-8)
        
        # Capacity Factor / Capacity Usage (논문 필수 지표)
        # Capacity factor: c * T/E where c is capacity factor, T is total tokens, E is num experts
        # Each expert can process up to capacity_factor * (total_tokens / num_experts) tokens
        if capacity_factor is None:
            # Default capacity factor (typically 1.0-2.0 in MoE papers)
            capacity_factor = 1.25
        ideal_capacity_per_expert = capacity_factor * (total_tokens / self.num_experts)
        max_expert_load = expert_token_counts.max().item()
        capacity_usage = max_expert_load / (ideal_capacity_per_expert + 1e-8)
        capacity_utilization = expert_token_counts.mean().item() / (ideal_capacity_per_expert + 1e-8)
        
        # Routing Sparsity / Number of experts activated per token (논문 필수 지표)
        # Average number of experts activated per token (top_k)
        avg_experts_per_token = float(top_k)
        routing_sparsity = 1.0 - (avg_experts_per_token / self.num_experts)  # Higher = more sparse
        
        # Fraction of active experts / Sparsity ratio (논문 필수 지표)
        # Fraction of experts that are actually used (have at least one token)
        fraction_active_experts = expert_utilization_rate  # Already computed above
        sparsity_ratio = 1.0 - fraction_active_experts  # Higher = more sparse (fewer experts used)
        
        # Expert Choice vs Token Choice routing metrics
        # Token choice: each token chooses experts (current implementation)
        # Expert choice: each expert chooses tokens (alternative)
        # For token choice, we measure how evenly tokens are distributed
        token_choice_entropy = normalized_proportion_entropy  # Already computed
        # For expert choice, we would measure how evenly experts choose tokens
        # Since we use token choice, we report token choice metrics
        routing_type = "token_choice"  # Current implementation
        
        # Standard deviation of tokens per expert (for aux loss calculation)
        std_tokens_per_expert = token_counts_std
        
        # Gini Coefficient - 논문 필수 지표 (LPR paper에서 사용)
        # Load distribution inequality 측정 (0 = perfect equality, 1 = perfect inequality)
        # Formula: G = (2 * sum(i * sorted_loads[i])) / (n * sum(sorted_loads)) - (n + 1) / n
        # Simplified: G = 1 - 2 * sum(cumulative_proportion) / n
        expert_loads_sorted = torch.sort(expert_token_counts, descending=False)[0].cpu().numpy()
        total_load = expert_loads_sorted.sum()
        if total_load > 0:
            # Cumulative proportion
            cumulative_proportion = np.cumsum(expert_loads_sorted) / total_load
            # Gini coefficient
            n = len(expert_loads_sorted)
            gini_coefficient = 1.0 - 2.0 * cumulative_proportion.sum() / (n + 1e-8)
        else:
            gini_coefficient = 0.0
        
        # Min-max Expert Load Ratio - 논문 필수 지표 (LPR paper에서 사용)
        # min(expert_load) / max(expert_load) (높을수록 균형, 1.0 = perfect balance)
        min_expert_load = expert_token_counts.min().item()
        max_expert_load = expert_token_counts.max().item()
        min_max_expert_load_ratio = min_expert_load / (max_expert_load + 1e-8)
        
        return {
            'expert_token_counts': expert_token_counts.cpu().numpy().tolist(),
            'expert_weighted_counts': expert_weighted_counts.cpu().numpy().tolist(),
            'expert_activation_counts': expert_activation_counts.cpu().numpy().tolist(),
            'load_balancing_coefficient': load_balancing_coefficient,
            'weighted_load_balancing_coefficient': weighted_load_balancing_coefficient,
            'load_imbalance_ratio': load_imbalance_ratio,
            'weighted_load_imbalance_ratio': weighted_load_imbalance_ratio,
            'expert_utilization_rate': expert_utilization_rate,
            'expert_token_proportions': expert_token_proportions,
            'expert_proportion_entropy': normalized_proportion_entropy,
            'expert_avg_routing_weight_mean': expert_avg_weight_mean,
            'expert_avg_routing_weight_std': expert_avg_weight_std,
            'total_tokens_processed': int(total_tokens),
            # Additional metrics from recent papers
            'maxvio': maxvio,
            'normalized_maxvio': normalized_maxvio,
            'aux_loss': aux_loss,
            'lpr': lpr,
            'expert_capacity_utilization': expert_capacity_utilization,
            'load_variance': load_variance,
            'expert_efficiency': expert_efficiency,
            # Additional metrics from user requirements
            'capacity_factor': capacity_factor,
            'ideal_capacity_per_expert': ideal_capacity_per_expert,
            'capacity_usage': capacity_usage,
            'capacity_utilization': capacity_utilization,
            'routing_sparsity': routing_sparsity,
            'avg_experts_per_token': avg_experts_per_token,
            'fraction_active_experts': fraction_active_experts,
            'sparsity_ratio': sparsity_ratio,
            'routing_type': routing_type,
            'token_choice_entropy': token_choice_entropy,
            'std_tokens_per_expert': std_tokens_per_expert,
            # 논문 필수 지표 추가
            'gini_coefficient': gini_coefficient,
            'min_max_expert_load_ratio': min_max_expert_load_ratio,
        }
    
    def _compute_inference_cost_metrics(
        self,
        selected_experts: torch.Tensor,  # [batch*seq, top_k]
        routing_weights: torch.Tensor,  # [batch*seq, top_k]
    ) -> Dict[str, float]:
        """
        Inference Cost / FLOPs / Utilized Parameters Metrics (논문 필수 지표)
        
        Measures:
        1. Number of active experts per token (routing sparsity)
        2. Total number of unique experts activated
        3. Expert activation ratio (active experts / total experts)
        4. Average experts per token
        """
        batch_seq_len = selected_experts.size(0)
        top_k = selected_experts.size(1)
        
        # Unique experts activated
        unique_experts = torch.unique(selected_experts.flatten())
        num_active_experts = len(unique_experts)
        
        # Expert activation ratio
        expert_activation_ratio = num_active_experts / self.num_experts
        
        # Average experts per token (top_k)
        avg_experts_per_token = float(top_k)
        
        # Total expert activations (counting duplicates)
        total_expert_activations = batch_seq_len * top_k
        
        # Expert utilization efficiency
        # Ideal: all experts used equally
        ideal_activations_per_expert = total_expert_activations / self.num_experts
        actual_activations = torch.bincount(selected_experts.flatten(), minlength=self.num_experts).float()
        utilization_efficiency = 1.0 - (actual_activations.std() / (actual_activations.mean() + 1e-8))
        
        return {
            'num_active_experts': num_active_experts,
            'expert_activation_ratio': expert_activation_ratio,
            'avg_experts_per_token': avg_experts_per_token,
            'total_expert_activations': total_expert_activations,
            'utilization_efficiency': utilization_efficiency.item(),
            # Note: FLOPs 계산은 별도 measure_efficiency.py에서 수행
            # 여기서는 활성화 관련 지표만 제공
        }
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """전체 학습 과정에 대한 집계 지표 (논문용)"""
        if not self.gram_matrix_history:
            return {}
        
        metrics = {
            'avg_gram_orthogonality': np.mean(self.gram_matrix_history),
            'std_gram_orthogonality': np.std(self.gram_matrix_history),
            'avg_speciality_penalty': np.mean(self.speciality_penalty_history),
            'avg_cosine_similarity': np.mean(self.cosine_similarity_history),
        }
        
        # Load balancing 집계 지표
        if self.load_balancing_coefficient_history:
            metrics['avg_load_balancing_coefficient'] = np.mean(self.load_balancing_coefficient_history)
            metrics['std_load_balancing_coefficient'] = np.std(self.load_balancing_coefficient_history)
            metrics['final_load_balancing_coefficient'] = self.load_balancing_coefficient_history[-1]
        
        # Expert token counts 집계
        if self.expert_token_counts_history:
            # 모든 스텝의 expert token counts를 평균
            all_counts = np.array(self.expert_token_counts_history)  # [num_steps, num_experts]
            avg_expert_token_counts = all_counts.mean(axis=0).tolist()
            std_expert_token_counts = all_counts.std(axis=0).tolist()
            final_expert_token_counts = all_counts[-1].tolist()
            
            metrics['avg_expert_token_counts'] = avg_expert_token_counts
            metrics['std_expert_token_counts'] = std_expert_token_counts
            metrics['final_expert_token_counts'] = final_expert_token_counts
            
            # 최종 스텝의 load balancing 지표
            final_counts = torch.tensor(final_expert_token_counts)
            final_mean = final_counts.mean().item()
            final_std = final_counts.std().item()
            metrics['final_load_balancing_cv'] = final_std / (final_mean + 1e-8)
            metrics['final_load_imbalance_ratio'] = final_counts.max().item() / (final_mean + 1e-8)
        
        # MaxVio 집계
        if self.maxvio_history:
            metrics['avg_maxvio'] = np.mean(self.maxvio_history)
            metrics['max_maxvio'] = np.max(self.maxvio_history)
            metrics['final_maxvio'] = self.maxvio_history[-1]
        
        # Aux Loss 집계
        if self.aux_loss_history:
            metrics['avg_aux_loss'] = np.mean(self.aux_loss_history)
            metrics['final_aux_loss'] = self.aux_loss_history[-1]
        
        # LPR 집계
        if self.lpr_history:
            metrics['avg_lpr'] = np.mean(self.lpr_history)
            metrics['final_lpr'] = self.lpr_history[-1]
        
        # DeepSpeed MoE metrics 집계
        if self.deepspeed_metrics_history:
            metrics['avg_expert_efficiency'] = np.mean([m['expert_efficiency'] for m in self.deepspeed_metrics_history])
            metrics['avg_expert_capacity_utilization'] = np.mean([m['expert_capacity_utilization'] for m in self.deepspeed_metrics_history])
            metrics['avg_load_variance'] = np.mean([m['load_variance'] for m in self.deepspeed_metrics_history])
            metrics['final_expert_efficiency'] = self.deepspeed_metrics_history[-1]['expert_efficiency']
        
        # Expert Utilization Rate 집계
        if self.expert_utilization_rate_history:
            metrics['avg_expert_utilization_rate'] = np.mean(self.expert_utilization_rate_history)
            metrics['final_expert_utilization_rate'] = self.expert_utilization_rate_history[-1]
        elif self.expert_token_counts_history:
            # Fallback: 최종 expert token counts에서 계산
            final_counts = np.array(self.expert_token_counts_history[-1])
            metrics['expert_utilization_rate'] = (final_counts > 0).sum() / len(final_counts)
        
        # Expert Specialization 지표 집계
        if self.expert_diversity_score_history:
            metrics['avg_expert_diversity_score'] = np.mean(self.expert_diversity_score_history)
            metrics['final_expert_diversity_score'] = self.expert_diversity_score_history[-1]
        
        if self.expert_similarity_mean_history:
            metrics['avg_expert_similarity_mean'] = np.mean(self.expert_similarity_mean_history)
            metrics['final_expert_similarity_mean'] = self.expert_similarity_mean_history[-1]
        
        if self.expert_specialization_strength_history:
            metrics['avg_expert_specialization_strength'] = np.mean(self.expert_specialization_strength_history)
            metrics['final_expert_specialization_strength'] = self.expert_specialization_strength_history[-1]
        
        # 논문 필수 지표 집계
        if hasattr(self, 'expert_overlap_history') and self.expert_overlap_history:
            metrics['avg_expert_overlap'] = np.mean(self.expert_overlap_history)
            metrics['final_expert_overlap'] = self.expert_overlap_history[-1]
            metrics['min_expert_overlap'] = np.min(self.expert_overlap_history)
            metrics['max_expert_overlap'] = np.max(self.expert_overlap_history)
        
        if hasattr(self, 'gini_coefficient_history') and self.gini_coefficient_history:
            metrics['avg_gini_coefficient'] = np.mean(self.gini_coefficient_history)
            metrics['final_gini_coefficient'] = self.gini_coefficient_history[-1]
            metrics['min_gini_coefficient'] = np.min(self.gini_coefficient_history)
            metrics['max_gini_coefficient'] = np.max(self.gini_coefficient_history)
        
        if hasattr(self, 'min_max_expert_load_ratio_history') and self.min_max_expert_load_ratio_history:
            metrics['avg_min_max_expert_load_ratio'] = np.mean(self.min_max_expert_load_ratio_history)
            metrics['final_min_max_expert_load_ratio'] = self.min_max_expert_load_ratio_history[-1]
            metrics['min_min_max_expert_load_ratio'] = np.min(self.min_max_expert_load_ratio_history)
            metrics['max_min_max_expert_load_ratio'] = np.max(self.min_max_expert_load_ratio_history)
        
        # Capacity, Sparsity, Routing metrics 집계 (최신 스텝에서 가져옴)
        # 이들은 per-step metrics이므로 최신 값만 사용
        # 실제로는 per-layer metrics에서 집계해야 하지만, 여기서는 간단히 처리
        
        return metrics
    
    def save_analysis(self, filepath: str):
        """분석 결과 저장 (논문용 데이터 포함)"""
        analysis_data = {
            'aggregated_metrics': self.get_aggregated_metrics(),
            'gram_matrix_history': self.gram_matrix_history,
            'speciality_penalty_history': self.speciality_penalty_history,
            'cosine_similarity_history': self.cosine_similarity_history,
            'load_balancing_coefficient_history': self.load_balancing_coefficient_history,
            'expert_token_counts_history': self.expert_token_counts_history,
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
    
    def get_paper_metrics_summary(self) -> Dict[str, Any]:
        """
        논문에 포함할 주요 지표 요약
        
        Returns:
            논문에 포함할 핵심 지표들 (데이터가 없으면 None 반환)
        """
        aggregated = self.get_aggregated_metrics()
        
        if not aggregated:
            return {}
        
        # 논문에 포함할 핵심 지표들
        # 데이터가 없으면 None 반환 (0.0으로 fallback하지 않음)
        summary = {
            # Load Balancing Metrics (필수)
            'load_balancing': {
                'coefficient_of_variation': aggregated.get('final_load_balancing_cv'),
                'load_imbalance_ratio': aggregated.get('final_load_imbalance_ratio'),
                'expert_utilization_rate': aggregated.get('expert_utilization_rate'),
                'expert_token_distribution': aggregated.get('final_expert_token_counts'),
                # 논문 필수 지표 추가
                'gini_coefficient': aggregated.get('final_gini_coefficient'),
                'min_max_expert_load_ratio': aggregated.get('final_min_max_expert_load_ratio'),
            },
            
            # Expert Specialization Metrics (필수)
            'expert_specialization': {
                'expert_diversity_score': aggregated.get('expert_diversity_score'),
                'expert_similarity_mean': aggregated.get('expert_similarity_mean'),
                'expert_specialization_strength': aggregated.get('expert_specialization_strength'),
                # 논문 필수 지표 추가
                'expert_overlap_mean': aggregated.get('final_expert_overlap'),
            },
            
            # Gram Matrix Quality
            'gram_matrix_quality': {
                'orthogonality': aggregated.get('avg_gram_orthogonality'),
                'orthogonality_std': aggregated.get('std_gram_orthogonality'),
            },
            
            # Routing Quality
            'routing_quality': {
                'routing_confidence': aggregated.get('routing_confidence'),
                'cosine_similarity_mean': aggregated.get('avg_cosine_similarity'),
            },
        }
        
        return summary

