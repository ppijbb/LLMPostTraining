# coding=utf-8
"""
Standard MoE Baseline (Switch Transformer style)

Implements a baseline MoE router using Switch Transformer's approach:
- Linear router (no GRU, no Expression projection)
- Standard load balancing auxiliary loss
- Top-k expert selection

This is used for fair comparison with SPECTRA MoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from tqdm.auto import tqdm

# Import necessary components
from models.spectra import (
    find_layers_in_model,
    find_mlp_in_layer,
    is_already_moe,
    copy_mlp_weights_to_expert,
    extract_config_info,
)
from .g3moe_model import G3MoEMLP


class SwitchRouter(nn.Module):
    """
    Switch Transformer style router.
    
    Simple linear router with load balancing auxiliary loss.
    No GRU, no Expression projection, no Gram matrix constraints.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        load_balance_loss_coef: float = 0.01,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.load_balance_loss_coef = load_balance_loss_coef
        
        # Simple linear router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x, hn=None, top_k=2, jitter_eps=0.01, training=True):
        """
        Forward pass for routing.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            hn: Hidden state (ignored for Switch router)
            top_k: Number of experts to select
            jitter_eps: Jitter noise for routing
            training: Whether in training mode
            
        Returns:
            multiplier: Routing weights [batch*seq, top_k]
            selected_experts: Selected expert indices [batch*seq, top_k]
            router_logits: Router logits (for compatibility)
            hn: Hidden state (None for Switch router)
            load_balance_loss: Load balancing auxiliary loss
            router_scores: Router scores for all experts [batch*seq, num_experts]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Flatten for routing
        x_flat = x.view(-1, hidden_size)  # [batch*seq, hidden_size]
        
        # Compute router logits
        router_logits = self.router(x_flat)  # [batch*seq, num_experts]
        
        # Add jitter noise during training
        if training and jitter_eps > 0:
            noise = torch.empty_like(router_logits).uniform_(-jitter_eps, jitter_eps)
            router_logits = router_logits + noise
        
        # Top-k expert selection
        top_k = min(top_k, self.num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        
        # Normalize top-k weights
        multiplier = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute load balancing auxiliary loss (Switch Transformer style)
        load_balance_loss = self._compute_load_balance_loss(routing_weights)
        
        # Return compatibility with SPECTRARouter
        return (
            multiplier,
            selected_experts,
            router_logits.view(batch_size, seq_len, self.num_experts),
            None,  # hn (not used for Switch router)
            load_balance_loss,
            routing_weights.view(batch_size, seq_len, self.num_experts),
            torch.tensor(0.0, device=x.device),  # expression_loss (not used)
        )
    
    def _compute_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss (Switch Transformer style).
        
        Loss = num_experts * sum(frac_i^2) where frac_i is the fraction of tokens routed to expert i.
        """
        # routing_weights: [batch*seq, num_experts]
        # Compute fraction of tokens per expert
        frac_per_expert = routing_weights.mean(dim=0)  # [num_experts]
        
        # Compute loss: num_experts * sum(frac_i^2)
        load_balance_loss = self.num_experts * (frac_per_expert ** 2).sum()
        
        return load_balance_loss * self.load_balance_loss_coef


class StandardMoEBlock(nn.Module):
    """
    Standard MoE Block using Switch Router.
    
    Similar structure to SPECTRABlock but uses SwitchRouter instead.
    """
    
    def __init__(
        self,
        router: SwitchRouter,
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
    ):
        super().__init__()
        self.hidden_dim = hidden_size or getattr(expert_config, 'hidden_size', 768)
        self.ffn_dim = self.hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = router
        self.expert_module_class = expert_module_class
        
        # Create experts
        self.experts = nn.ModuleList([
            expert_module_class(expert_config, intermediate_size=intermediate_size)
            for _ in range(self.num_experts)
        ])
        
        # Create shared experts
        shared_intermediate_size = (intermediate_size or getattr(expert_config, 'intermediate_size', 3072)) * n_shared_experts
        self.shared_experts = expert_module_class(
            expert_config,
            intermediate_size=shared_intermediate_size
        )
        
        if freeze_shared_experts:
            for param in self.shared_experts.parameters():
                param.requires_grad = False
        
        self.router_jitter_noise = router_jitter_noise
        self.input_jitter_noise = input_jitter_noise
        self.return_router_scores = False  # By default, return only hidden states
    
    def forward(self, hidden_states, global_routing_hn=None):
        """
        Forward pass.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_dim]
            global_routing_hn: Hidden state for routing (ignored for Switch router)
            
        Returns:
            hidden_states: Output hidden states
            router_scores: Router scores (if return_router_scores=True)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply input jitter noise during training
        if self.training and self.input_jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise
            )
        
        # Router forward
        (
            routing_weights,
            selected_experts,
            router_logits,
            _,
            load_balance_loss,
            router_scores,
            _,
        ) = self.router(
            hidden_states,
            hn=None,
            top_k=self.top_k,
            jitter_eps=self.router_jitter_noise if self.training else 0.0,
            training=self.training,
        )
        
        # Flatten for processing
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [batch*seq, hidden_dim]
        
        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # Process through experts
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)  # [batch*seq]
            
            if not expert_mask.any():
                continue
            
            # Get indices
            expert_indices = torch.where(expert_mask)[0]
            
            # Get routing weights for this expert
            expert_positions = []
            expert_weights = []
            for idx in expert_indices:
                # Find which position in top_k this expert is at
                for k in range(self.top_k):
                    if selected_experts[idx, k] == expert_idx:
                        expert_positions.append(idx)
                        expert_weights.append(routing_weights[idx, k])
                        break
            
            if not expert_positions:
                continue
            
            expert_positions = torch.tensor(expert_positions, device=hidden_states.device)
            expert_weights = torch.stack(expert_weights).unsqueeze(-1)  # [num_tokens, 1]
            
            # Process through expert
            expert_input = hidden_states_flat[expert_positions]
            expert_output = self.experts[expert_idx](expert_input)
            
            # Weight and accumulate
            final_hidden_states[expert_positions] += expert_output * expert_weights
        
        # Process through shared experts
        shared_output = self.shared_experts(hidden_states_flat)
        final_hidden_states = final_hidden_states + shared_output
        
        # Reshape back
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        
        # Store load balance loss for later retrieval
        self._load_balance_loss = load_balance_loss
        
        if self.return_router_scores:
            return final_hidden_states, router_scores
        else:
            return final_hidden_states


@torch.no_grad()
def upcycle_to_switch_moe(
    model: nn.Module,
    moe_config: Dict[str, Any],
    expert_module_class: type = G3MoEMLP,
    layer_start_idx: int = 0,
    layer_end_idx: Optional[int] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Upcycle any pretrained model to Standard MoE (Switch Transformer style).
    
    This is used as a baseline for comparison with SPECTRA MoE.
    
    Args:
        model: The pretrained transformer model to upcycle
        moe_config: Dictionary containing MoE configuration
            Required keys:
                - hidden_size: Hidden dimension size
                - intermediate_size: Intermediate MLP dimension
                - num_experts: Number of experts
                - num_experts_per_tok: Top-k experts per token
                - n_shared_experts: Number of shared experts (default: 1)
                - first_k_dense_replace: First k layers to keep as dense (default: 0)
                - load_balance_loss_coef: Load balancing loss coefficient (default: 0.01)
        expert_module_class: Class for expert MLP modules (default: G3MoEMLP)
        layer_start_idx: Starting layer index for conversion (default: 0)
        layer_end_idx: Ending layer index for conversion (None = all layers)
        verbose: Whether to show progress bar
        
    Returns:
        Model with MoE layers using SwitchRouter (modified in-place, also returns the model)
    """
    # Extract configuration
    cfg = extract_config_info(moe_config)
    
    # Detect model dtype
    model_dtype = None
    try:
        for param in model.parameters():
            if param is not None:
                model_dtype = param.dtype
                break
        if model_dtype is None and hasattr(model, 'config'):
            if hasattr(model.config, 'torch_dtype'):
                model_dtype = model.config.torch_dtype
    except Exception:
        pass
    
    if model_dtype is None:
        try:
            for buffer in model.buffers():
                if buffer is not None:
                    model_dtype = buffer.dtype
                    break
        except Exception:
            pass
    
    if model_dtype is None:
        if verbose:
            print("Warning: Could not detect model dtype, defaulting to float32")
        model_dtype = torch.float32
    
    if verbose:
        print(f"Detected model dtype: {model_dtype}")
    
    # Find layers
    layers = find_layers_in_model(model)
    if layers is None:
        raise ValueError(
            "Could not find decoder layers in model. "
            "Please ensure the model has 'layers', 'h', 'block', or 'decoder_layers' attribute."
        )
    
    # Create Switch router (shared across all layers)
    router = SwitchRouter(
        hidden_size=cfg['hidden_size'],
        num_experts=cfg['num_experts'],
        load_balance_loss_coef=cfg.get('load_balance_loss_coef', 0.01),
    )
    router = router.to(dtype=model_dtype)
    
    # Determine layer range
    if layer_end_idx is None:
        layer_end_idx = len(layers)
    else:
        layer_end_idx = min(layer_end_idx, len(layers))
    
    layer_range = range(layer_start_idx, layer_end_idx)
    
    if verbose:
        processing = tqdm(
            enumerate(layer_range),
            total=len(layer_range),
            desc=f"Upcycling model to Standard MoE",
            leave=False
        )
    else:
        processing = enumerate(layer_range)
    
    for enum_idx, layer_idx in processing:
        decoder_layer = layers[layer_idx]
        
        # Find MLP in layer
        source_mlp = find_mlp_in_layer(decoder_layer)
        if source_mlp is None:
            if verbose:
                processing.set_description(f"Layer {layer_idx}: No MLP found, skipping")
            continue
        
        # Skip if before first_k_dense_replace
        if layer_idx < cfg['first_k_dense_replace']:
            if verbose:
                processing.set_description(f"Layer {layer_idx}: Before first_k_dense_replace, skipping")
            continue
        
        # Skip if already MoE (for baseline comparison, we only convert dense models)
        if is_already_moe(source_mlp):
            if verbose:
                processing.set_description(f"Layer {layer_idx}: Already MoE, skipping (baseline only converts dense models)")
            continue
        
        # Create MoE block
        if verbose:
            processing.set_description(f"Layer {layer_idx}: Creating Standard MoE block")
        
        # Create dummy config
        class DummyConfig:
            def __init__(self):
                self.hidden_size = cfg['hidden_size']
                self.intermediate_size = cfg['intermediate_size']
                self.hidden_activation = cfg['hidden_activation']
        
        expert_config = DummyConfig()
        
        moe_block = StandardMoEBlock(
            router=router,
            expert_module_class=expert_module_class,
            expert_config=expert_config,
            num_experts=cfg['num_experts'],
            top_k=cfg['num_experts_per_tok'],
            n_shared_experts=cfg['n_shared_experts'],
            hidden_size=cfg['hidden_size'],
            intermediate_size=cfg['intermediate_size'],
            router_jitter_noise=cfg['router_jitter_noise'],
            input_jitter_noise=cfg['input_jitter_noise'],
            freeze_shared_experts=cfg['freeze_shared_experts'],
        )
        moe_block = moe_block.to(dtype=model_dtype)
        
        # Replace MLP with MoE block
        for attr_name in ['mlp', 'feed_forward', 'ffn', 'ffw', 'moe']:
            if hasattr(decoder_layer, attr_name):
                setattr(decoder_layer, attr_name, moe_block)
                break
        
        # Copy weights from source MLP to experts
        if verbose:
            processing.set_description(f"Layer {layer_idx}: Copying weights to experts")
        
        # Copy to shared experts
        if hasattr(moe_block, 'shared_experts'):
            copy_mlp_weights_to_expert(source_mlp, moe_block.shared_experts)
        
        # Copy to routed experts
        if hasattr(moe_block, 'experts'):
            for expert_idx, expert in enumerate(moe_block.experts):
                if expert_idx % 10 == 0 and verbose:
                    processing.set_description(f"Layer {layer_idx}: Copying to expert {expert_idx}/{len(moe_block.experts)}")
                copy_mlp_weights_to_expert(source_mlp, expert)
    
    if verbose:
        if isinstance(processing, tqdm):
            processing.set_description("Upcycling completed")
    
    return model


__all__ = [
    'SwitchRouter',
    'StandardMoEBlock',
    'upcycle_to_switch_moe',
]

