# coding=utf-8
"""
Expert Specialization Analysis for MoE Models

Provides comprehensive analysis of expert specialization including:
- Expert affinity scores per domain
- Domain purity metrics
- Token clustering quality (Silhouette score)
- Expert diversity (NMI)
- Routing confidence
- Representation similarity analysis
- Visualization tools (t-SNE, heatmaps, PCA)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from tqdm.auto import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# ML metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Save results
import json
import pickle
from pathlib import Path


@dataclass
class ExpertFeatures:
    """Expert에서 추출한 features를 저장하는 데이터 클래스"""
    embeddings: np.ndarray  # [num_tokens, hidden_dim]
    expert_ids: np.ndarray  # [num_tokens, top_k]
    routing_weights: np.ndarray  # [num_tokens, top_k]
    domain_labels: np.ndarray  # [num_tokens]
    token_texts: List[str]  # [num_tokens]
    
    def __len__(self):
        return len(self.embeddings)


class ExpertSpecializationAnalyzer:
    """
    MoE 모델의 Expert Specialization을 종합적으로 분석하는 클래스
    
    주요 기능:
    1. Feature 추출
    2. Specialization 메트릭 계산
    3. 시각화 생성
    4. 리포트 작성
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = 'cuda',
        domain_keywords: Optional[Dict[str, List[str]]] = None
    ):
        """
        Args:
            model: MoE 모델 (SPECTRA, Switch, 등)
            tokenizer: Tokenizer
            device: Device ('cuda' or 'cpu')
            domain_keywords: Domain별 키워드 (자동 domain labeling용)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.domain_keywords = domain_keywords or self._default_domain_keywords()
        
        # Model을 eval mode로
        self.model.eval()
        
    def _default_domain_keywords(self) -> Dict[str, List[str]]:
        """기본 domain keyword dict"""
        return {
            'code': ['def', 'function', 'class', 'import', 'return', 'if', 'for', 'while', '()', '{}'],
            'math': ['equation', 'solve', 'calculate', '+', '-', '*', '/', '=', 'x', 'y', 'integral'],
            'science': ['experiment', 'hypothesis', 'theory', 'data', 'research', 'study', 'analysis'],
            'history': ['century', 'war', 'period', 'era', 'ancient', 'medieval', 'revolution'],
            'literature': ['character', 'novel', 'story', 'author', 'book', 'write', 'poem', 'narrative'],
            'business': ['market', 'economy', 'company', 'profit', 'investment', 'strategy', 'management'],
            'technology': ['system', 'computer', 'data', 'network', 'software', 'hardware', 'algorithm'],
        }
    
    def _detect_domain(self, text: str) -> str:
        """텍스트에서 domain 자동 탐지"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'general'
    
    @torch.no_grad()
    def extract_features(
        self,
        dataloader: Any,
        max_samples: int = 10000,
        layer_idx: int = -1
    ) -> ExpertFeatures:
        """
        모델에서 features 추출
        
        Args:
            dataloader: 데이터 로더
            max_samples: 최대 샘플 수
            layer_idx: Hidden state를 추출할 layer index (-1 = last layer)
            
        Returns:
            ExpertFeatures 객체
        """
        all_embeddings = []
        all_expert_ids = []
        all_routing_weights = []
        all_domain_labels = []
        all_token_texts = []
        
        num_samples = 0
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            if num_samples >= max_samples:
                break
            
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract hidden states (last layer before expert)
            hidden_states = outputs.hidden_states[layer_idx]  # [batch, seq, dim]
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Extract expert assignments and routing weights
            # MoE 레이어에서 routing 정보 추출 (모델마다 다를 수 있음)
            expert_ids, routing_weights = self._extract_routing_info(outputs)
            
            # Convert to numpy
            embeddings = hidden_states.cpu().numpy().reshape(-1, hidden_dim)
            expert_ids_np = expert_ids.cpu().numpy().reshape(-1, expert_ids.shape[-1])
            routing_weights_np = routing_weights.cpu().numpy().reshape(-1, routing_weights.shape[-1])
            
            # Extract token texts for qualitative analysis
            token_texts = []
            for i in range(batch_size):
                tokens = input_ids[i].cpu().numpy()
                texts = [self.tokenizer.decode([t]) for t in tokens]
                token_texts.extend(texts)
            
            # Detect domains (can be replaced with ground truth if available)
            domain_labels = np.array([self._detect_domain(text) for text in token_texts])
            
            # Append
            all_embeddings.append(embeddings)
            all_expert_ids.append(expert_ids_np)
            all_routing_weights.append(routing_weights_np)
            all_domain_labels.append(domain_labels)
            all_token_texts.extend(token_texts)
            
            num_samples += batch_size * seq_len
        
        # Concatenate
        features = ExpertFeatures(
            embeddings=np.vstack(all_embeddings),
            expert_ids=np.vstack(all_expert_ids),
            routing_weights=np.vstack(all_routing_weights),
            domain_labels=np.concatenate(all_domain_labels),
            token_texts=all_token_texts
        )
        
        return features
    
    def _extract_routing_info(self, outputs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """모델 outputs에서 routing 정보 추출"""
        # SPECTRA 스타일
        if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
            # router_logits: tuple of [batch*seq, num_experts]
            router_logits = outputs.router_logits
            if isinstance(router_logits, tuple):
                # Use last layer's routing
                router_logits = router_logits[-1]
            
            # Top-k selection
            routing_probs = F.softmax(router_logits, dim=-1)
            routing_weights, expert_ids = torch.topk(routing_probs, k=2, dim=-1)
            
            return expert_ids, routing_weights
        
        # Fallback: dummy routing (모델이 routing 정보를 제공하지 않는 경우)
        else:
            batch_size, seq_len = outputs.last_hidden_state.shape[:2]
            num_experts = 8  # Default
            expert_ids = torch.randint(0, num_experts, (batch_size, seq_len, 2))
            routing_weights = torch.rand(batch_size, seq_len, 2)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            return expert_ids, routing_weights
    
    # ===========================
    # Metric Computation
    # ===========================
    
    def compute_affinity_matrix(self, features: ExpertFeatures) -> np.ndarray:
        """
        Expert-domain affinity matrix 계산
        
        Affinity(expert_i, domain_d) = P(expert_i | domain_d) / P(expert_i)
        
        Returns:
            affinity_matrix: [num_experts, num_domains]
        """
        num_experts = features.expert_ids.max() + 1
        domains = list(set(features.domain_labels))
        num_domains = len(domains)
        domain_to_idx = {d: i for i, d in enumerate(domains)}
        
        affinity_matrix = np.zeros((num_experts, num_domains))
        
        # Compute P(expert_i)
        expert_counts = np.bincount(features.expert_ids[:, 0], minlength=num_experts)
        p_expert = expert_counts / expert_counts.sum()
        
        # Compute P(expert_i | domain_d)
        for domain_idx, domain in enumerate(domains):
            domain_mask = features.domain_labels == domain
            if domain_mask.sum() == 0:
                continue
            
            domain_expert_ids = features.expert_ids[domain_mask, 0]
            domain_expert_counts = np.bincount(domain_expert_ids, minlength=num_experts)
            p_expert_given_domain = domain_expert_counts / domain_expert_counts.sum()
            
            # Affinity = P(expert | domain) / P(expert)
            affinity = p_expert_given_domain / (p_expert + 1e-8)
            affinity_matrix[:, domain_idx] = affinity
        
        return affinity_matrix, domains
    
    def compute_domain_purity(self, features: ExpertFeatures) -> Dict[str, float]:
        """
        각 expert의 domain purity 계산
        
        Purity(expert_i) = max_d (count(domain_d, expert_i) / total_count(expert_i))
        
        Returns:
            dict: {expert_id: purity_score}
        """
        num_experts = features.expert_ids.max() + 1
        purity_scores = {}
        
        for expert_id in range(num_experts):
            mask = features.expert_ids[:, 0] == expert_id
            if mask.sum() == 0:
                purity_scores[expert_id] = 0.0
                continue
            
            expert_domains = features.domain_labels[mask]
            domain_counts = {}
            for domain in expert_domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            max_count = max(domain_counts.values())
            total_count = sum(domain_counts.values())
            purity = max_count / total_count
            purity_scores[expert_id] = purity
        
        return purity_scores
    
    def compute_silhouette_score(self, features: ExpertFeatures, sample_size: int = 10000) -> float:
        """
        Token clustering quality 측정 (Silhouette score)
        
        Args:
            features: ExpertFeatures
            sample_size: 계산 속도를 위한 샘플링 크기
            
        Returns:
            silhouette_score: [-1, 1]
        """
        # Sample for computational efficiency
        if len(features) > sample_size:
            indices = np.random.choice(len(features), sample_size, replace=False)
            embeddings = features.embeddings[indices]
            labels = features.expert_ids[indices, 0]
        else:
            embeddings = features.embeddings
            labels = features.expert_ids[:, 0]
        
        # Compute silhouette score
        score = silhouette_score(embeddings, labels, metric='cosine', sample_size=min(5000, len(embeddings)))
        
        return float(score)
    
    def compute_expert_diversity(self, features: ExpertFeatures) -> float:
        """
        Expert 간 diversity 측정 (Normalized Mutual Information)
        
        Returns:
            diversity: [0, 1], higher is more diverse
        """
        num_experts = features.expert_ids.max() + 1
        
        # Compute NMI between all expert pairs
        nmi_scores = []
        for i in range(num_experts):
            for j in range(i+1, num_experts):
                mask_i = features.expert_ids[:, 0] == i
                mask_j = features.expert_ids[:, 0] == j
                
                if mask_i.sum() == 0 or mask_j.sum() == 0:
                    continue
                
                # Domain distributions for expert i and j
                domains_i = features.domain_labels[mask_i]
                domains_j = features.domain_labels[mask_j]
                
                # Compute NMI (requires same length, so we sample)
                min_len = min(len(domains_i), len(domains_j))
                if min_len < 10:
                    continue
                
                domains_i_sample = np.random.choice(domains_i, min_len, replace=False)
                domains_j_sample = np.random.choice(domains_j, min_len, replace=False)
                
                nmi = normalized_mutual_info_score(domains_i_sample, domains_j_sample)
                nmi_scores.append(nmi)
        
        # Diversity = 1 - mean(NMI)
        diversity = 1.0 - np.mean(nmi_scores) if nmi_scores else 0.0
        
        return float(diversity)
    
    def compute_routing_confidence(self, features: ExpertFeatures) -> float:
        """
        Routing confidence 측정 (Gini coefficient 기반)
        
        Confidence = 1 - Gini = 1 - (1 - sum(p_i^2))
        
        Returns:
            confidence: [0, 1], higher is more confident
        """
        # Gini coefficient
        gini_scores = []
        for weights in features.routing_weights:
            # Normalize
            weights = weights / weights.sum()
            # Gini = 1 - sum(p^2)
            gini = 1.0 - np.sum(weights ** 2)
            gini_scores.append(gini)
        
        # Confidence = 1 - mean(Gini)
        confidence = 1.0 - np.mean(gini_scores)
        
        return float(confidence)
    
    def compute_representation_similarity(self, features: ExpertFeatures) -> np.ndarray:
        """
        Expert 간 representation similarity matrix 계산
        
        Returns:
            similarity_matrix: [num_experts, num_experts]
        """
        num_experts = features.expert_ids.max() + 1
        
        # Compute average embedding per expert
        expert_representations = []
        for expert_id in range(num_experts):
            mask = features.expert_ids[:, 0] == expert_id
            if mask.sum() == 0:
                expert_representations.append(np.zeros(features.embeddings.shape[1]))
            else:
                expert_representations.append(features.embeddings[mask].mean(axis=0))
        
        expert_representations = np.array(expert_representations)
        
        # Compute cosine similarity matrix
        similarity_matrix = np.zeros((num_experts, num_experts))
        for i in range(num_experts):
            for j in range(num_experts):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = np.dot(expert_representations[i], expert_representations[j])
                    sim /= (np.linalg.norm(expert_representations[i]) * np.linalg.norm(expert_representations[j]) + 1e-8)
                    similarity_matrix[i, j] = sim
        
        return similarity_matrix
    
    def extract_expert_keywords(
        self,
        features: ExpertFeatures,
        top_k: int = 20
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        TF-IDF로 각 expert의 특징적 키워드 추출
        
        Returns:
            dict: {expert_id: [(keyword, score), ...]}
        """
        num_experts = features.expert_ids.max() + 1
        expert_keywords = {}
        
        for expert_id in range(num_experts):
            mask = features.expert_ids[:, 0] == expert_id
            if mask.sum() == 0:
                expert_keywords[expert_id] = []
                continue
            
            expert_texts = [features.token_texts[i] for i in range(len(mask)) if mask[i]]
            
            # Skip if too few samples
            if len(expert_texts) < 10:
                expert_keywords[expert_id] = []
                continue
            
            # TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform(expert_texts)
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.sum(axis=0).A1
                
                # Top keywords
                top_indices = scores.argsort()[-top_k:][::-1]
                keywords = [(feature_names[i], float(scores[i])) for i in top_indices]
                expert_keywords[expert_id] = keywords
            except:
                expert_keywords[expert_id] = []
        
        return expert_keywords
    
    # ===========================
    # Visualization
    # ===========================
    
    def visualize_token_clustering(
        self,
        features: ExpertFeatures,
        save_path: str,
        method: str = 'tsne',
        sample_size: int = 5000
    ):
        """t-SNE 또는 UMAP으로 token clustering 시각화"""
        # Sample for computational efficiency
        if len(features) > sample_size:
            indices = np.random.choice(len(features), sample_size, replace=False)
            embeddings = features.embeddings[indices]
            expert_ids = features.expert_ids[indices, 0]
            domain_labels = features.domain_labels[indices]
        else:
            embeddings = features.embeddings
            expert_ids = features.expert_ids[:, 0]
            domain_labels = features.domain_labels
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # By expert
        num_experts = expert_ids.max() + 1
        for expert_id in range(num_experts):
            mask = expert_ids == expert_id
            ax1.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=f'Expert {expert_id}',
                alpha=0.6,
                s=10
            )
        ax1.set_title(f'Token Clustering by Expert ({method.upper()})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(alpha=0.3)
        
        # By domain
        domains = list(set(domain_labels))
        for domain in domains:
            mask = domain_labels == domain
            ax2.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=domain,
                alpha=0.6,
                s=10
            )
        ax2.set_title(f'Token Clustering by Domain ({method.upper()})')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_affinity_heatmap(
        self,
        affinity_matrix: np.ndarray,
        domains: List[str],
        save_path: str
    ):
        """Expert-domain affinity heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            affinity_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            xticklabels=domains,
            yticklabels=[f'E{i}' for i in range(affinity_matrix.shape[0])],
            cbar_kws={'label': 'Affinity Score'},
            vmin=0,
            vmax=4
        )
        plt.title('Expert-Domain Affinity Matrix\n(Higher = More Specialized)', fontsize=14)
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('Expert', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_expert_representations(
        self,
        features: ExpertFeatures,
        save_path: str
    ):
        """Expert representation space PCA 시각화"""
        num_experts = features.expert_ids.max() + 1
        
        # Compute expert representations
        expert_reprs = []
        for expert_id in range(num_experts):
            mask = features.expert_ids[:, 0] == expert_id
            if mask.sum() == 0:
                expert_reprs.append(np.zeros(features.embeddings.shape[1]))
            else:
                expert_reprs.append(features.embeddings[mask].mean(axis=0))
        expert_reprs = np.array(expert_reprs)
        
        # PCA
        pca = PCA(n_components=2)
        expert_reprs_2d = pca.fit_transform(expert_reprs)
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(expert_reprs_2d[:, 0], expert_reprs_2d[:, 1], s=200, c='blue', alpha=0.6)
        
        for i in range(num_experts):
            plt.annotate(
                f'E{i}',
                (expert_reprs_2d[i, 0], expert_reprs_2d[i, 1]),
                fontsize=12,
                ha='center'
            )
            # Draw vector from origin
            plt.arrow(
                0, 0,
                expert_reprs_2d[i, 0], expert_reprs_2d[i, 1],
                alpha=0.3,
                width=0.01,
                head_width=0.05,
                color='red'
            )
        
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(alpha=0.3)
        plt.title('Expert Representation Space (PCA)\nVectors from origin show expert directions', fontsize=14)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_domain_purity(
        self,
        purity_scores: Dict[int, float],
        save_path: str
    ):
        """Domain purity bar chart"""
        expert_ids = sorted(purity_scores.keys())
        purities = [purity_scores[i] for i in expert_ids]
        
        plt.figure(figsize=(10, 6))
        plt.bar(expert_ids, purities, color='skyblue', edgecolor='black')
        plt.axhline(1/7, color='red', linestyle='--', label='Uniform (no specialization)')
        plt.xlabel('Expert ID', fontsize=12)
        plt.ylabel('Domain Purity', fontsize=12)
        plt.title('Domain Purity per Expert\n(Higher = More Specialized)', fontsize=14)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # ===========================
    # Report Generation
    # ===========================
    
    def generate_report(
        self,
        features: ExpertFeatures,
        save_dir: str,
        model_name: str = 'model'
    ):
        """
        전체 분석 리포트 생성
        
        생성되는 파일:
        - metrics.json: 모든 메트릭
        - affinity_heatmap.pdf
        - token_clustering_tsne.pdf
        - expert_representations_pca.pdf
        - domain_purity_bar.pdf
        - expert_keywords.json
        - report.txt: 텍스트 리포트
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Generating Expert Specialization Report for {model_name}")
        print(f"{'='*60}\n")
        
        # Compute all metrics
        print("Computing metrics...")
        affinity_matrix, domains = self.compute_affinity_matrix(features)
        purity_scores = self.compute_domain_purity(features)
        silhouette = self.compute_silhouette_score(features)
        diversity = self.compute_expert_diversity(features)
        confidence = self.compute_routing_confidence(features)
        repr_similarity = self.compute_representation_similarity(features)
        expert_keywords = self.extract_expert_keywords(features)
        
        # Save metrics
        metrics = {
            'model_name': model_name,
            'num_tokens': len(features),
            'num_experts': int(features.expert_ids.max() + 1),
            'affinity_scores': {
                'mean': float(affinity_matrix[affinity_matrix != 1.0].mean()),
                'max': float(affinity_matrix.max()),
                'matrix': affinity_matrix.tolist()
            },
            'domain_purity': {
                'mean': float(np.mean(list(purity_scores.values()))),
                'per_expert': {int(k): float(v) for k, v in purity_scores.items()}
            },
            'silhouette_score': float(silhouette),
            'expert_diversity': float(diversity),
            'routing_confidence': float(confidence),
            'representation_similarity': {
                'mean_off_diagonal': float(repr_similarity[np.triu_indices_from(repr_similarity, k=1)].mean()),
                'matrix': repr_similarity.tolist()
            }
        }
        
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Metrics saved to {save_dir / 'metrics.json'}")
        
        # Save keywords
        with open(save_dir / 'expert_keywords.json', 'w') as f:
            json.dump(expert_keywords, f, indent=2)
        
        print(f"✓ Keywords saved to {save_dir / 'expert_keywords.json'}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        self.visualize_affinity_heatmap(
            affinity_matrix, domains,
            str(save_dir / 'affinity_heatmap.pdf')
        )
        print(f"✓ Affinity heatmap saved")
        
        self.visualize_token_clustering(
            features,
            str(save_dir / 'token_clustering_tsne.pdf'),
            method='tsne'
        )
        print(f"✓ Token clustering (t-SNE) saved")
        
        self.visualize_expert_representations(
            features,
            str(save_dir / 'expert_representations_pca.pdf')
        )
        print(f"✓ Expert representations (PCA) saved")
        
        self.visualize_domain_purity(
            purity_scores,
            str(save_dir / 'domain_purity_bar.pdf')
        )
        print(f"✓ Domain purity bar chart saved")
        
        # Generate text report
        print("\nGenerating text report...")
        report_lines = [
            f"Expert Specialization Analysis Report",
            f"=" * 60,
            f"Model: {model_name}",
            f"Number of tokens analyzed: {len(features):,}",
            f"Number of experts: {features.expert_ids.max() + 1}",
            f"",
            f"Summary Metrics:",
            f"-" * 60,
            f"Average Affinity Score:       {metrics['affinity_scores']['mean']:.3f}",
            f"Average Domain Purity:        {metrics['domain_purity']['mean']:.3f}",
            f"Silhouette Score:             {metrics['silhouette_score']:.3f}",
            f"Expert Diversity (NMI):       {metrics['expert_diversity']:.3f}",
            f"Routing Confidence:           {metrics['routing_confidence']:.3f}",
            f"Repr. Similarity (off-diag):  {metrics['representation_similarity']['mean_off_diagonal']:.3f}",
            f"",
            f"Per-Expert Analysis:",
            f"-" * 60,
        ]
        
        for expert_id in sorted(purity_scores.keys()):
            report_lines.append(f"\nExpert {expert_id}:")
            report_lines.append(f"  Domain Purity: {purity_scores[expert_id]:.3f}")
            
            if expert_id in expert_keywords and expert_keywords[expert_id]:
                top_keywords = expert_keywords[expert_id][:5]
                keywords_str = ", ".join([k for k, _ in top_keywords])
                report_lines.append(f"  Top Keywords: {keywords_str}")
        
        report_text = "\n".join(report_lines)
        
        with open(save_dir / 'report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"✓ Text report saved to {save_dir / 'report.txt'}")
        
        print(f"\n{'='*60}")
        print(f"Report generation complete!")
        print(f"All files saved to: {save_dir}")
        print(f"{'='*60}\n")
        
        return metrics


def compare_models(
    analyzer1: ExpertSpecializationAnalyzer,
    analyzer2: ExpertSpecializationAnalyzer,
    dataloader: Any,
    save_dir: str,
    model1_name: str = 'Model1',
    model2_name: str = 'Model2',
    max_samples: int = 10000
):
    """두 모델의 expert specialization 비교"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract features
    print(f"Extracting features from {model1_name}...")
    features1 = analyzer1.extract_features(dataloader, max_samples=max_samples)
    
    print(f"Extracting features from {model2_name}...")
    features2 = analyzer2.extract_features(dataloader, max_samples=max_samples)
    
    # Generate individual reports
    print(f"\nGenerating report for {model1_name}...")
    metrics1 = analyzer1.generate_report(features1, str(save_dir / model1_name), model1_name)
    
    print(f"\nGenerating report for {model2_name}...")
    metrics2 = analyzer2.generate_report(features2, str(save_dir / model2_name), model2_name)
    
    # Comparison report
    print("\nGenerating comparison report...")
    comparison = {
        'model1': model1_name,
        'model2': model2_name,
        'comparison': {
            'affinity_score': {
                'model1': metrics1['affinity_scores']['mean'],
                'model2': metrics2['affinity_scores']['mean'],
                'improvement': (metrics2['affinity_scores']['mean'] - metrics1['affinity_scores']['mean']) / metrics1['affinity_scores']['mean'] * 100
            },
            'domain_purity': {
                'model1': metrics1['domain_purity']['mean'],
                'model2': metrics2['domain_purity']['mean'],
                'improvement': (metrics2['domain_purity']['mean'] - metrics1['domain_purity']['mean']) / metrics1['domain_purity']['mean'] * 100
            },
            'silhouette_score': {
                'model1': metrics1['silhouette_score'],
                'model2': metrics2['silhouette_score'],
                'improvement': (metrics2['silhouette_score'] - metrics1['silhouette_score']) / abs(metrics1['silhouette_score']) * 100
            },
            'expert_diversity': {
                'model1': metrics1['expert_diversity'],
                'model2': metrics2['expert_diversity'],
                'improvement': (metrics2['expert_diversity'] - metrics1['expert_diversity']) / metrics1['expert_diversity'] * 100
            }
        }
    }
    
    with open(save_dir / 'comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"✓ Comparison saved to {save_dir / 'comparison.json'}")
    print("\nComparison Summary:")
    print(f"{'Metric':<25} {model1_name:<15} {model2_name:<15} {'Improvement':<15}")
    print("-" * 70)
    for metric, values in comparison['comparison'].items():
        print(f"{metric:<25} {values['model1']:<15.3f} {values['model2']:<15.3f} {values['improvement']:+.1f}%")
    
    return comparison

