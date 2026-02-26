import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from models.seqorth_model import SeqorthRouter
from models.seqorth_config import SeqorthTextConfig

def test_router_checkpointing():
    print("Testing SeqorthRouter with Gradient Checkpointing...")
    
    # Mock config
    config = SeqorthTextConfig(
        hidden_size=576,
        n_routed_experts=32,
        router_dim=128,
        router_impl="seqorth"
    )
    
    router = SeqorthRouter(config).cuda().bfloat16()
    router.train()
    
    # Input
    batch_size = 2
    seq_len = 2048
    x = torch.randn(batch_size, seq_len, 576, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    hn = None # Will be initialized inside
    
    # Mock forward with checkpointing
    def forward_wrapper(input_tensor):
        # We need to wrap it because checkpoint expects a function that takes only tensors
        # and returns only tensors.
        # router.forward(x, hn, top_k=2, jitter_eps=0.01, step_frac=0.0)
        output = router(input_tensor, None)
        # router returns (routing_probs_full, routing_output_flat, hn_next, router_loss_dict)
        # We only care about the first one for simplicity or just return all flattened
        return output[0] # routing_probs_full
    
    try:
        print("Running forward/backward with checkpointing...")
        # Checkpoint expects (function, *args)
        # use_reentrant=True is common but can be tricky
        probs = checkpoint(forward_wrapper, x, use_reentrant=True)
        
        loss = probs.sum()
        loss.backward()
        
        print("Success! Backward pass completed without RuntimeError.")
        
        # Check if gradients exist
        if x.grad is not None:
            print(f"Input grad shape: {x.grad.shape}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_router_checkpointing()
