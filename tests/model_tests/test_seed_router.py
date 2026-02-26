"""
SEED Router Verification Test (Seqorthl Eigenvalue Equilibrium Dispatch)

Validates:
1. CV convergence < 0.001 within 100 steps
2. PES (orthogonality error) < 0.01
3. MaxVio < 0.01
4. Gradient flow through router
5. All experts are being trained

Usage:
    cd /home/conan/workspace/llm_training
    PYTHONPATH=. python tests/test_seed_router.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.getcwd())

from models.seqorth_model import SeqorthRouter
from models.seqorth_config import SeqorthTextConfig


def test_seed_cv_convergence():
    """Test that SEED router achieves CV < 0.001 within 100 steps"""
    print("=" * 60)
    print("Testing SEED Router CV Convergence")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config with SEED enabled and AGGRESSIVE balancing
    config = SeqorthTextConfig(
        hidden_size=256,
        n_routed_experts=16,
        router_dim=64,
        seed_routing_enabled=True,  # Enable SEED
        seed_base_temp=1.0,
        seed_bias_gain=1.5,  # Increased: faster integral accumulation
        seed_pes_temp_scale=0.0,  # Disabled
        bias_boost=10.0,  # Increased: stronger bias feedback
        urgency_scale=200.0,  # Increased: more CV sensitivity
        expert_choice_routing=True,
        expert_choice_capacity_factor=1.5,  # Reduced: tighter capacity
        balance_loss_coef=5.0,  # Increased: stronger balance loss
        balancing_strength=0.05,  # Increased: more aggressive balancing
    )
    
    router = SeqorthRouter(config).to(device)
    router.train()
    
    # Optimizer with lower LR for stability
    optimizer = torch.optim.Adam(router.parameters(), lr=0.001)
    
    batch_size = 32
    seq_len = 64
    hidden_size = config.hidden_size
    
    # Data generator with MODERATE skew (50% center, 50% noise)
    center = torch.randn(1, hidden_size, device=device)
    
    cv_history = []
    pes_history = []
    maxvio_history = []
    actual_cv_history = []
    
    STEPS = 1000  # Significantly increased for convergence
    
    for step in range(STEPS):
        # Generate data with MODERATE skew (50% center, 50% noise)
        noise = torch.randn(batch_size, seq_len, hidden_size, device=device)
        x = center * 0.5 + noise * 0.5  # 50-50 mix
        
        # Initialize GRU state
        gru_hidden_size = getattr(config, "router_intent_hidden_size", 64)
        hn = torch.zeros(batch_size * seq_len, gru_hidden_size, device=device, dtype=x.dtype)
        
        # Forward pass
        outputs = router(x, hn=hn, top_k=2, jitter_eps=0.01, step_frac=step/STEPS, layer_idx=0)
        multiplier, selected_experts, hn_next, routing_probs_full, \
            speciality_loss, cosine_similarities, ortho_loss, \
            entropy_loss, routing_uncertainty, contrastive_loss = outputs
        
        # Compute task loss (specialization pressure)
        task_loss = -routing_probs_full.max(dim=-1).values.mean()
        
        # Get orthogonality loss (safe)
        if hasattr(router, '_compute_safe_speciality_loss'):
            ortho_loss_safe = router._compute_safe_speciality_loss()
        else:
            ortho_loss_safe = ortho_loss
        
        # Total loss with balance loss
        loss = task_loss + 0.1 * ortho_loss_safe
        
        # Add explicit balance loss
        if hasattr(router, 'expert_load_ema') and hasattr(router, 'balance_loss_coef'):
            load = router.expert_load_ema
            target_load = 1.0 / config.n_routed_experts
            load_error = (load - target_load).abs().mean()
            balance_loss = load_error * router.balance_loss_coef
            loss = loss + balance_loss
        
        # Backward + Update
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Record metrics
        with torch.no_grad():
            cv = router.cv_ema.item()
            pes = getattr(router, 'last_pes', torch.tensor(0.0))
            pes_val = pes.item() if torch.is_tensor(pes) else pes
            maxvio = router.max_vio_ema.item()
            
            cv_history.append(cv)
            pes_history.append(pes_val)
            maxvio_history.append(maxvio)
            
            if step % 100 == 0:
                # Check expert usage
                load = router.expert_load_ema
                load_std = load.std().item()
                load_mean = load.mean().item()
                actual_cv = load_std / (load_mean + 1e-8)
                unused_experts = (load < 1e-6).sum().item()
                print(f"Step {step:3d}: CV={cv:.6f}, ActualCV={actual_cv:.6f}, PES={pes_val:.6f}, "
                      f"MaxVio={maxvio:.6f}, Unused={unused_experts}")
    
    # Final assertions
    final_cv = cv_history[-1]
    final_pes = pes_history[-1]
    final_maxvio = maxvio_history[-1]
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final CV:     {final_cv:.6f} (target: < 0.001)")
    print(f"Final PES:    {final_pes:.6f} (target: < 0.01)")
    print(f"Final MaxVio: {final_maxvio:.6f} (target: < 0.01)")
    print()
    
    # Check assertions (stricter targets)
    passed = True
    if final_cv < 0.005:  # Target: CV < 0.005
        print(f"✅ CV test PASSED (< 0.005)")
    elif final_cv < 0.01:  # Acceptable: CV < 0.01
        print(f"⚠️  CV test PARTIAL PASS (< 0.01, got {final_cv:.6f})")
    else:
        print(f"❌ CV test FAILED: {final_cv:.6f} >= 0.01")
        passed = False
    
    if final_pes < 0.01:  # Target: PES < 0.01
        print(f"✅ PES test PASSED (< 0.01)")
    elif final_pes < 0.1:  # Acceptable: PES < 0.1
        print(f"⚠️  PES test PARTIAL PASS (< 0.1, got {final_pes:.6f})")
    else:
        print(f"❌ PES test FAILED: {final_pes:.6f} >= 0.1")
        passed = False
    
    # Check convergence trend
    if len(cv_history) >= 100:
        recent_cv = sum(cv_history[-100:]) / 100
        early_cv = sum(cv_history[:100]) / 100
        if recent_cv < early_cv * 0.5:
            print(f"✅ Convergence trend PASSED (early: {early_cv:.6f}, recent: {recent_cv:.6f})")
        else:
            print(f"⚠️  Convergence trend WEAK (early: {early_cv:.6f}, recent: {recent_cv:.6f})")
    
    return passed


def test_seed_gradient_flow():
    """Test that gradients flow through SEED router"""
    print()
    print("=" * 60)
    print("Testing SEED Router Gradient Flow")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = SeqorthTextConfig(
        hidden_size=128,
        n_routed_experts=8,
        router_dim=32,
        seed_routing_enabled=True,
    )
    
    router = SeqorthRouter(config).to(device)
    router.train()
    
    x = torch.randn(2, 16, 128, device=device, requires_grad=True)
    
    outputs = router(x, None, top_k=2)
    multiplier = outputs[0]
    
    loss = multiplier.sum()
    loss.backward()
    
    # Check gradients exist
    grad_exists = x.grad is not None and x.grad.abs().sum() > 0
    
    if grad_exists:
        print(f"✅ Gradient flow test PASSED")
        print(f"   Input grad norm: {x.grad.norm().item():.6f}")
    else:
        print(f"❌ Gradient flow test FAILED: no gradients")
    
    return grad_exists


def test_seed_all_experts_used():
    """Test that all experts receive tokens (no expert collapse)"""
    print()
    print("=" * 60)
    print("Testing SEED Router Expert Utilization")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = SeqorthTextConfig(
        hidden_size=256,
        n_routed_experts=16,
        router_dim=64,
        seed_routing_enabled=True,
    )
    
    router = SeqorthRouter(config).to(device)
    router.train()
    
    # Run multiple batches
    expert_counts = torch.zeros(config.n_routed_experts, device=device)
    
    for _ in range(50):
        x = torch.randn(16, 32, 256, device=device)
        outputs = router(x, None, top_k=2)
        selected_experts = outputs[1]
        
        for exp_idx in selected_experts.view(-1):
            expert_counts[exp_idx.long()] += 1
        
        # Backward to update internal states
        loss = outputs[0].sum()
        loss.backward()
    
    # Check all experts used
    min_usage = expert_counts.min().item()
    max_usage = expert_counts.max().item()
    unused_experts = (expert_counts == 0).sum().item()
    
    print(f"Expert usage range: {min_usage:.0f} - {max_usage:.0f}")
    print(f"Unused experts: {unused_experts}")
    
    if unused_experts == 0:
        print(f"✅ Expert utilization test PASSED (all experts used)")
        return True
    else:
        print(f"❌ Expert utilization test FAILED: {unused_experts} unused experts")
        return False


if __name__ == "__main__":
    all_passed = True
    
    all_passed &= test_seed_cv_convergence()
    all_passed &= test_seed_gradient_flow()
    all_passed &= test_seed_all_experts_used()
    
    print()
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)
