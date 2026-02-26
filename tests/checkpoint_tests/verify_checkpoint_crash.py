import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.seqorth_model import SeqorthRouter, SeqorthMoE
from models.seqorth_config import SeqorthTextConfig

class DummyExpert(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        return self.net(x)

def verify_checkpoint_fix():
    print("Testing SeqorthMoE Checkpoint Fix...")
    
    config = SeqorthTextConfig()
    config.hidden_size = 128
    config.n_routed_experts = 4
    config.router_dim = 16
    config.expert_choice_routing = True
    config.capacity_factor = 1.25
    config.balancing_strength = 0.1
    config.num_experts_per_tok = 2
    
    # Setup global router
    router = SeqorthRouter(config)
    
    # Setup experts
    experts = nn.ModuleList([DummyExpert(config.hidden_size) for _ in range(config.n_routed_experts)])
    
    # Setup MoE
    moe = SeqorthMoE(config, experts, router=router)
    moe.training = True
    moe.train()
    
    # Hack to allow registering router (if typically handled by model)
    # Testing direct usage logic
    
    batch_size = 4
    seq_len = 32
    hidden_size = 128
    
    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    hn = None
    
    # Capture initial states
    initial_bias = router.expert_bias.clone()
    initial_ema = moe.expert_specialization_ema.clone()
    initial_max_vio = router.max_vio_ema.clone()
    
    print(f"Initial Bias Norm: {initial_bias.norm().item()}")

    def run_forward(hidden_states):
        # Wrapper to match signature expected by checkpoint
        # SeqorthMoE forward: (hidden_states, hn_state)
        return moe(hidden_states)

    print("\nRunning Checkpointed Forward + Backward (use_reentrant=False)...")
    try:
        # Run forward with checkpointing
        # Note: SeqorthMoE has _gradient_checkpointing_func override usually,
        # but here we call checkpoint manually on a wrapper or the module itself.
        # But wait, SeqorthMoE disables checkpointing internally via override!
        # If we wrap it in checkpoint, the override is bypassed because we call checkpoint directly.
        # checkpoint(run_forward, x) calls run_forward inside no_grad (reentrant=False doesn't use no_grad?)
        # With use_reentrant=False, it runs forward, saves inputs.
        
        output = checkpoint(run_forward, x, use_reentrant=False, preserve_rng_state=True)
        # output is (layer_output, (router_outs...))
        
        layer_output = output[0]
        # router_outs = output[1]
        
        loss = layer_output.mean()
        
        print("Forward complete. Running Backward...")
        loss.backward()
        print("Backward complete.")
        
    except Exception as e:
        print(f"\n❌ CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check updates
    final_bias = router.expert_bias
    bias_diff = (final_bias - initial_bias).norm().item()
    print(f"\nBias Update Diff: {bias_diff}")
    
    final_ema = moe.expert_specialization_ema
    ema_diff = (final_ema - initial_ema).norm().item()
    print(f"Specialization EMA Diff: {ema_diff}")

    final_max_vio = router.max_vio_ema
    max_vio_diff = (final_max_vio - initial_max_vio).item()
    print(f"MaxVio EMA Diff: {max_vio_diff}")
    
    if bias_diff > 0 or ema_diff > 0 or abs(max_vio_diff) > 0:
        print("✅ SUCCESS: Updates applied (via backward hook) and Checkpointing passed!")
    else:
        print("⚠️ WARNING: No updates applied. Hook might not have run or gradients zero?")
        
    print("Verification complete.")

if __name__ == "__main__":
    verify_checkpoint_fix()
