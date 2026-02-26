import torch
import torch.nn as nn
from models.seqorth_model import SeqorthRouter, ExpressionProjector
from models.seqorth_config import SeqorthTextConfig

def verify_router_fix():
    print("Testing SeqorthRouter fix...")
    
    config = SeqorthTextConfig()
    config.hidden_size = 128
    config.n_routed_experts = 8
    config.router_dim = 16
    config.expert_choice_routing = True
    config.capacity_factor = 1.25
    
    router = SeqorthRouter(config)
    router.training = True
    
    batch_size = 4
    seq_len = 32
    hidden_size = 128
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    hn = None
    
    # Run forward pass multiple times to see if CV and ortho loss behave well
    for i in range(5):
        outputs = router(x, hn, step_frac=i/5.0)
        multiplier, selected_experts, hn_next, routing_probs_full, speciality_loss, domain_orthogonality, ortho_loss, entropy_loss, routing_uncertainty, contrastive_loss = outputs
        
        # Check CV
        cv = router.cv_ema.item()
        
        # Check orthogonality of expression weight
        # W has shape [E*R, H]. To check expert orthogonality as used in logic:
        # We look at the projection weight rearranged
        W = router.expression.exp_proj.weight # [E*R, H]
        W_expert = W.view(config.n_routed_experts, config.router_dim, config.hidden_size)
        # Average representation over dimension (as done in forward: current_expert_repr)
        # Or just check if W @ W.T is identity for the projector itself.
        # The Newton-Schulz and Stiefel retraction work on W [E*R, H].
        Gram = torch.matmul(W, W.t()) # [E*R, E*R]
        Ortho_err = (Gram - torch.eye(config.n_routed_experts * config.router_dim, device=W.device, dtype=W.dtype)).norm().item()
        
        print(f"Step {i}: CV={cv:.4f}, OrthoErr={Ortho_err:.6f}, speciality_loss={speciality_loss.item():.6e}, ortho_loss={ortho_loss.item():.6e}")
        
    print("Verification complete.")

if __name__ == "__main__":
    verify_router_fix()
