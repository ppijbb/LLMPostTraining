# coding=utf-8
"""
Information-Theoretic Analysis for Seqorth MoE

핵심 질문:
1. 각 expert가 입력의 어떤 정보를 담당하는가? (Mutual Information)
2. Seqorth MoE가 정보를 어떻게 압축하고 보존하는가? (Information Bottleneck)
3. Expert space가 얼마나 많은 정보를 담을 수 있는가? (Representation Capacity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import math


class MutualInformationEstimator:
    """
    Mutual Information 추정기
    
    Methods:
    - MINE (Mutual Information Neural Estimation)
    - KDE-based estimation
    - Histogram-based estimation
    """
    
    def __init__(self, method: str = "kde", bins: int = 50):
        self.method = method
        self.bins = bins
    
    def estimate_mi(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Estimate I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
        
        Args:
            x: [N, ...] tensor
            y: [N, ...] tensor
            
        Returns:
            Mutual information estimate
        """
        if self.method == "kde":
            return self._estimate_mi_kde(x, y)
        elif self.method == "histogram":
            return self._estimate_mi_histogram(x, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _estimate_mi_kde(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """KDE-based MI estimation"""
        # Flatten to 1D for simplicity
        x_flat = x.flatten().cpu().numpy()
        y_flat = y.flatten().cpu().numpy()
        
        # Use scipy if available, otherwise fallback to histogram
        try:
            from scipy.stats import gaussian_kde
            from scipy.integrate import dblquad
            
            # Estimate joint and marginal densities
            xy = np.vstack([x_flat, y_flat])
            kde_xy = gaussian_kde(xy)
            kde_x = gaussian_kde(x_flat)
            kde_y = gaussian_kde(y_flat)
            
            # MI = E[log(p(x,y) / (p(x)p(y)))]
            # Approximate using samples
            n_samples = min(1000, len(x_flat))
            indices = np.random.choice(len(x_flat), n_samples, replace=False)
            
            mi_samples = []
            for idx in indices:
                x_val, y_val = x_flat[idx], y_flat[idx]
                try:
                    p_xy = kde_xy([x_val, y_val])[0]
                    p_x = kde_x([x_val])[0]
                    p_y = kde_y([y_val])[0]
                    if p_xy > 0 and p_x > 0 and p_y > 0:
                        mi_samples.append(np.log(p_xy / (p_x * p_y)))
                except:
                    continue
            
            return np.mean(mi_samples) if mi_samples else 0.0
        except ImportError:
            # Fallback to histogram
            return self._estimate_mi_histogram(x, y)
    
    def _estimate_mi_histogram(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Histogram-based MI estimation"""
        x_flat = x.flatten().cpu().numpy()
        y_flat = y.flatten().cpu().numpy()
        
        # Create 2D histogram
        hist_xy, x_edges, y_edges = np.histogram2d(x_flat, y_flat, bins=self.bins)
        hist_x, _ = np.histogram(x_flat, bins=x_edges)
        hist_y, _ = np.histogram(y_flat, bins=y_edges)
        
        # Normalize to probabilities
        p_xy = hist_xy / (hist_xy.sum() + 1e-10)
        p_x = hist_x / (hist_x.sum() + 1e-10)
        p_y = hist_y / (hist_y.sum() + 1e-10)
        
        # Compute MI = sum(p(x,y) * log(p(x,y) / (p(x)p(y))))
        mi = 0.0
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                if p_xy[i, j] > 0:
                    p_x_val = p_x[i] if i < len(p_x) else 0.0
                    p_y_val = p_y[j] if j < len(p_y) else 0.0
                    if p_x_val > 0 and p_y_val > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x_val * p_y_val + 1e-10))
        
        return max(0.0, mi)


class InformationTheoreticAnalyzer:
    """
    Information-theoretic analysis for Seqorth MoE
    
    핵심 분석:
    1. Expert-Input Mutual Information
    2. Expert-Expert Mutual Information
    3. Information Bottleneck Analysis
    4. Representation Capacity Analysis
    """
    
    def __init__(self, num_experts: int, mi_estimator: Optional[MutualInformationEstimator] = None):
        self.num_experts = num_experts
        self.mi_estimator = mi_estimator or MutualInformationEstimator()
        self.reset()
    
    def reset(self):
        """Reset analysis state"""
        self.expert_input_mi = defaultdict(list)
        self.expert_expert_mi = defaultdict(dict)
        self.layer_information_flow = []
        self.representation_capacity = []
    
    def analyze_expert_input_mi(
        self,
        expert_activations: Dict[int, torch.Tensor],  # expert_idx -> [N, hidden_dim]
        input_representations: torch.Tensor,  # [N, hidden_dim]
    ) -> Dict[int, float]:
        """
        Compute mutual information between each expert and input.
        
        Returns:
            Dictionary mapping expert_idx -> MI(Expert; Input)
        """
        mi_scores = {}
        
        for expert_idx, expert_outputs in expert_activations.items():
            # Flatten for MI computation
            expert_flat = expert_outputs.flatten(start_dim=1)  # [N, ...]
            input_flat = input_representations.flatten(start_dim=1)  # [N, ...]
            
            # Compute MI for each dimension (or use PCA for dimensionality reduction)
            # For efficiency, use mean activation
            expert_mean = expert_flat.mean(dim=1)  # [N]
            input_mean = input_flat.mean(dim=1)  # [N]
            
            mi = self.mi_estimator.estimate_mi(expert_mean, input_mean)
            mi_scores[expert_idx] = mi
            
            self.expert_input_mi[expert_idx].append(mi)
        
        return mi_scores
    
    def analyze_expert_expert_mi(
        self,
        expert_activations: Dict[int, torch.Tensor],
    ) -> np.ndarray:
        """
        Compute pairwise mutual information between experts.
        
        Returns:
            MI matrix [num_experts, num_experts]
        """
        mi_matrix = np.zeros((self.num_experts, self.num_experts))
        
        expert_indices = sorted(expert_activations.keys())
        
        for i, expert_i_idx in enumerate(expert_indices):
            for j, expert_j_idx in enumerate(expert_indices):
                if i == j:
                    mi_matrix[i, j] = 1.0  # Self-MI (normalized)
                else:
                    expert_i = expert_activations[expert_i_idx]
                    expert_j = expert_activations[expert_j_idx]
                    
                    # Flatten and compute mean
                    expert_i_flat = expert_i.flatten(start_dim=1).mean(dim=1)
                    expert_j_flat = expert_j.flatten(start_dim=1).mean(dim=1)
                    
                    mi = self.mi_estimator.estimate_mi(expert_i_flat, expert_j_flat)
                    mi_matrix[i, j] = mi
        
        # Store for analysis
        for i, expert_i_idx in enumerate(expert_indices):
            for j, expert_j_idx in enumerate(expert_indices):
                if i != j:
                    if expert_i_idx not in self.expert_expert_mi:
                        self.expert_expert_mi[expert_i_idx] = {}
                    self.expert_expert_mi[expert_i_idx][expert_j_idx] = mi_matrix[i, j]
        
        return mi_matrix
    
    def analyze_information_bottleneck(
        self,
        layer_hidden_states: List[torch.Tensor],  # [layer_0, layer_1, ..., layer_L]
        input_representations: torch.Tensor,
        task_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Information Bottleneck Analysis: I(X; Z_l) and I(Z_l; Y)
        
        Args:
            layer_hidden_states: Hidden states at each layer
            input_representations: Input representations
            task_labels: Optional task labels for I(Z_l; Y)
            
        Returns:
            Dictionary with information bottleneck metrics
        """
        results = {
            'compression': [],  # I(X; Z_l) / H(X)
            'relevance': [],    # I(Z_l; Y) / I(X; Y) if Y available
            'information_preservation': [],  # I(X; Z_l)
        }
        
        # Compute input entropy (approximate)
        input_flat = input_representations.flatten(start_dim=1)
        input_entropy = self._estimate_entropy(input_flat.mean(dim=1))
        
        for layer_idx, hidden_states in enumerate(layer_hidden_states):
            hidden_flat = hidden_states.flatten(start_dim=1)
            hidden_mean = hidden_flat.mean(dim=1)
            input_mean = input_flat.mean(dim=1)
            
            # I(X; Z_l)
            i_x_z = self.mi_estimator.estimate_mi(input_mean, hidden_mean)
            compression = i_x_z / (input_entropy + 1e-10)
            
            results['compression'].append(compression)
            results['information_preservation'].append(i_x_z)
            
            # I(Z_l; Y) if task labels available
            if task_labels is not None:
                i_z_y = self.mi_estimator.estimate_mi(hidden_mean, task_labels.float())
                # I(X; Y) for normalization
                i_x_y = self.mi_estimator.estimate_mi(input_mean, task_labels.float())
                relevance = i_z_y / (i_x_y + 1e-10)
                results['relevance'].append(relevance)
        
        self.layer_information_flow.append(results)
        return results
    
    def analyze_representation_capacity(
        self,
        expert_representations: torch.Tensor,  # [num_experts, representation_dim]
        gram_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Analyze representation capacity of expert space.
        
        Metrics:
        - Effective dimension
        - Rank analysis
        - Information capacity: log(det(Gram_matrix))
        """
        results = {}
        
        # Effective dimension (using SVD)
        U, S, V = torch.linalg.svd(expert_representations, full_matrices=False)
        # Effective dimension = number of significant singular values
        threshold = S.max() * 1e-3
        effective_dim = (S > threshold).sum().item()
        results['effective_dimension'] = effective_dim
        results['full_dimension'] = expert_representations.shape[1]
        results['dimension_ratio'] = effective_dim / expert_representations.shape[1]
        
        # Rank
        results['rank'] = torch.linalg.matrix_rank(expert_representations).item()
        
        # Information capacity from Gram matrix
        if gram_matrix is not None:
            try:
                # log(det(G)) = sum(log(eigenvalues))
                eigenvals = torch.linalg.eigvalsh(gram_matrix)
                eigenvals = eigenvals[eigenvals > 1e-8]  # Filter near-zero
                log_det = eigenvals.log().sum().item()
                results['information_capacity'] = log_det
                results['gram_matrix_determinant'] = eigenvals.prod().item()
            except:
                results['information_capacity'] = 0.0
                results['gram_matrix_determinant'] = 0.0
        
        # Orthogonality-capacity relationship
        if gram_matrix is not None:
            identity = torch.eye(gram_matrix.shape[0], device=gram_matrix.device)
            orthogonality = 1.0 - torch.norm(gram_matrix - identity, p='fro') / (gram_matrix.shape[0] * np.sqrt(2))
            results['orthogonality'] = orthogonality.item()
            results['orthogonality_capacity_correlation'] = orthogonality * results.get('information_capacity', 0.0)
        
        self.representation_capacity.append(results)
        return results
    
    def _estimate_entropy(self, x: torch.Tensor) -> float:
        """Estimate entropy H(X)"""
        x_flat = x.flatten().cpu().numpy()
        hist, _ = np.histogram(x_flat, bins=50)
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return entropy
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated information-theoretic metrics"""
        aggregated = {
            'expert_input_mi': {},
            'expert_expert_mi_mean': {},
            'information_bottleneck_summary': {},
            'representation_capacity_summary': {},
        }
        
        # Expert-Input MI summary
        for expert_idx, mi_list in self.expert_input_mi.items():
            if mi_list:
                aggregated['expert_input_mi'][expert_idx] = {
                    'mean': np.mean(mi_list),
                    'std': np.std(mi_list),
                    'min': np.min(mi_list),
                    'max': np.max(mi_list),
                }
        
        # Expert-Expert MI summary
        for expert_i, expert_j_dict in self.expert_expert_mi.items():
            if expert_j_dict:
                mi_values = list(expert_j_dict.values())
                aggregated['expert_expert_mi_mean'][expert_i] = np.mean(mi_values)
        
        # Information bottleneck summary
        if self.layer_information_flow:
            all_compression = [r['compression'] for r in self.layer_information_flow]
            all_preservation = [r['information_preservation'] for r in self.layer_information_flow]
            
            aggregated['information_bottleneck_summary'] = {
                'mean_compression': np.mean([np.mean(c) for c in all_compression]),
                'compression_trend': [np.mean(c) for c in all_compression],
                'mean_preservation': np.mean([np.mean(p) for p in all_preservation]),
            }
        
        # Representation capacity summary
        if self.representation_capacity:
            aggregated['representation_capacity_summary'] = {
                'mean_effective_dimension': np.mean([r.get('effective_dimension', 0) for r in self.representation_capacity]),
                'mean_rank': np.mean([r.get('rank', 0) for r in self.representation_capacity]),
                'mean_information_capacity': np.mean([r.get('information_capacity', 0) for r in self.representation_capacity]),
            }
        
        return aggregated


__all__ = [
    'MutualInformationEstimator',
    'InformationTheoreticAnalyzer',
]

