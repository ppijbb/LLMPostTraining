# coding=utf-8
"""
Seqorth Ablation Variants

Creates ablation variants of Seqorth MoE by removing/modifying components:
1. -Expression: Remove expression projector
2. -GRU: Replace GRU with linear layer
3. -SpecialityPenalty: Remove speciality penalty
4. -OrthoConstraint: Remove orthogonal constraint
5. StandardRouter: Use Switch-style router instead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from models.seqorth import SeqorthRouter, SeqorthBlock
from models.g3moe_model import ExpressionProjector


class AblationRouter(nn.Module):
    """
    Base class for ablation variants of SeqorthRouter.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        router_dim: int = 128,
        balancing_strength: float = 0.01,
        ema_alpha: float = 0.99,
        ablation_type: str = "none",
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router_dim = router_dim
        self.balancing_strength = balancing_strength
        self.ema_alpha = ema_alpha
        self.ablation_type = ablation_type
        self.register_buffer("expert_load_ema", torch.zeros(self.num_experts), persistent=True)
        
        # GRU or Linear routing
        if ablation_type == "no_gru":
            # Use linear layer instead of GRU
            self.load_balancer = nn.Linear(hidden_size, num_experts * router_dim, bias=False)
        else:
            # Standard GRU
            self.load_balancer = nn.GRU(
                input_size=hidden_size,
                hidden_size=num_experts * router_dim,
                num_layers=1,
                bias=False,
                batch_first=True,
            )
        
        # Expression projector (can be removed)
        if ablation_type != "no_expression":
            self.expression_projector = ExpressionProjector(
                hidden_size,
                router_dim,
                num_experts,
                method='precomputed'
            )
        else:
            self.expression_projector = None
    
    def forward(self, x, hn, top_k=2, jitter_eps=0.01, training=True):
        """
        Forward pass with ablation variants.
        """
        top_k = min(top_k, self.num_experts)
        
        batch_size, seq_len, hidden_size = x.shape
        input_shape = (batch_size, seq_len)
        hidden_shape = (*input_shape, -1, self.router_dim)
        
        # Routing logits (GRU or Linear)
        if self.ablation_type == "no_gru":
            # Linear routing
            x_flat = x.view(-1, hidden_size)  # [batch*seq, hidden_size]
            routing_logits_flat = self.load_balancer(x_flat)  # [batch*seq, num_experts * router_dim]
            routing_logits = routing_logits_flat.view(batch_size * seq_len, self.num_experts, self.router_dim)
            hn = None  # No hidden state for linear
        else:
            # GRU routing
            routing_logits, hn = self.load_balancer(x, hn.to(x.dtype) if hn is not None else None)
            routing_logits = routing_logits.view(batch_size * seq_len, self.num_experts, self.router_dim)
            hn = hn if hn is not None else torch.zeros(1, batch_size, self.num_experts * self.router_dim, device=x.device, dtype=x.dtype)
        
        routing_logits = F.normalize(routing_logits, dim=-1)
        
        # Expression projection (if not ablated)
        if self.ablation_type == "no_expression":
            # Use routing logits as expression logits
            expression_logits = routing_logits
        else:
            expression_logits = self.expression_projector(x)
            expression_logits = expression_logits.view(batch_size * seq_len, self.num_experts, self.router_dim)
        
        # Gram matrix and speciality penalty (if not ablated)
        if self.ablation_type == "no_ortho" or self.ablation_type == "no_penalty":
            speciality_penalty = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        else:
            routing_logits_reshaped = routing_logits.view(batch_size, seq_len, self.num_experts, self.router_dim)
            gram = torch.matmul(routing_logits_reshaped, routing_logits_reshaped.transpose(-2, -1))
            routing_i = torch.eye(self.num_experts, device=routing_logits.device, dtype=routing_logits.dtype)
            routing_i_expanded = routing_i.unsqueeze(0).unsqueeze(0)
            
            if self.ablation_type == "no_penalty":
                speciality_penalty = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            else:
                speciality_penalty = torch.mean((F.normalize(gram - routing_i_expanded, dim=-1) ** 2).sum(dim=(-2, -1)))
        
        # Cosine similarity
        cosine_similarities = 1.0 - F.cosine_similarity(expression_logits, routing_logits, dim=-1)
        
        # Domain scores (with or without speciality penalty)
        if self.ablation_type == "no_penalty":
            domain_scores = cosine_similarities
        else:
            penalty_expanded = (1.0 + speciality_penalty.unsqueeze(-1).unsqueeze(-1))
            domain_scores = cosine_similarities * penalty_expanded
        
        # Top-k selection
        domain_scores_flat = domain_scores.view(batch_size * seq_len, self.num_experts)
        top_k_logits, selected_experts = torch.topk(domain_scores_flat, top_k, dim=-1)
        multiplier = F.softmax(top_k_logits, dim=-1)
        
        # Expression loss (if not ablated)
        if self.ablation_type == "no_expression":
            expression_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        else:
            expression_loss = self.expression_projector.orthogonal_loss()
        
        # Load balancing (same as original)
        if self.training:
            with torch.no_grad():
                total_load = self.expert_load_ema.sum()
                if total_load > 0:
                    load_balancing_scores = self.expert_load_ema / total_load
                else:
                    load_balancing_scores = torch.zeros_like(self.expert_load_ema)
                
                adjustment = load_balancing_scores * self.balancing_strength * self.num_experts
                multiplier = multiplier - adjustment[selected_experts]
                
                current_load = F.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=[0, 1]).float()
                self.expert_load_ema.mul_(self.ema_alpha).add_(current_load, alpha=1.0 - self.ema_alpha)
        
        # Reshape for compatibility
        cosine_similarities = cosine_similarities.view(batch_size, seq_len, self.num_experts)
        
        return multiplier, selected_experts, expression_logits, hn, speciality_penalty, cosine_similarities, expression_loss


def create_ablation_router(
    hidden_size: int,
    num_experts: int,
    router_dim: int = 128,
    balancing_strength: float = 0.01,
    ema_alpha: float = 0.99,
    ablation_type: str = "none",
) -> nn.Module:
    """
    Create an ablation variant router.
    
    Args:
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        router_dim: Router dimension
        balancing_strength: Load balancing strength
        ema_alpha: EMA alpha for load balancing
        ablation_type: Type of ablation
            - "none": Full Seqorth (no ablation)
            - "no_expression": Remove expression projector
            - "no_gru": Replace GRU with linear layer
            - "no_penalty": Remove speciality penalty
            - "no_ortho": Remove orthogonal constraint
            - "standard_router": Use Switch-style router (from standard_moe_upcycle)
    
    Returns:
        Router module (SeqorthRouter, AblationRouter, or SwitchRouter)
    """
    if ablation_type == "none":
        # Full Seqorth
        return SeqorthRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_dim=router_dim,
            balancing_strength=balancing_strength,
            ema_alpha=ema_alpha,
        )
    elif ablation_type == "standard_router":
        # Switch-style router
        from models.standard_moe_upcycle import SwitchRouter
        return SwitchRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            load_balance_loss_coef=balancing_strength,
        )
    else:
        # Ablation variant
        return AblationRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            router_dim=router_dim,
            balancing_strength=balancing_strength,
            ema_alpha=ema_alpha,
            ablation_type=ablation_type,
        )


def create_ablation_moe_block(
    router: nn.Module,
    expert_module_class: type,
    expert_config: Any,
    num_experts: int,
    top_k: int = 2,
    n_shared_experts: int = 1,
    hidden_size: int = None,
    intermediate_size: int = None,
    router_jitter_noise: float = 0.01,
    input_jitter_noise: float = 0.0,
    freeze_shared_experts: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create an ablation variant MoE block.
    
    Uses StandardMoEBlock if router is SwitchRouter, otherwise uses SeqorthBlock.
    """
    if isinstance(router, nn.Module) and hasattr(router, 'ablation_type') and router.ablation_type == "standard_router":
        from models.standard_moe_upcycle import StandardMoEBlock
        return StandardMoEBlock(
            router=router,
            expert_module_class=expert_module_class,
            expert_config=expert_config,
            num_experts=num_experts,
            top_k=top_k,
            n_shared_experts=n_shared_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            router_jitter_noise=router_jitter_noise,
            input_jitter_noise=input_jitter_noise,
            freeze_shared_experts=freeze_shared_experts,
        )
    else:
        return SeqorthBlock(
            router=router,
            expert_module_class=expert_module_class,
            expert_config=expert_config,
            num_experts=num_experts,
            top_k=top_k,
            n_shared_experts=n_shared_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            router_jitter_noise=router_jitter_noise,
            input_jitter_noise=input_jitter_noise,
            freeze_shared_experts=freeze_shared_experts,
        )


__all__ = [
    'AblationRouter',
    'create_ablation_router',
    'create_ablation_moe_block',
]

