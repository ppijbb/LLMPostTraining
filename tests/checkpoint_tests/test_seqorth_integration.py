#!/usr/bin/env python3
"""
SpecHorn Integration Test
Tests that all SpecHorn components are properly integrated
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from models.seqorth_model import SeqorthRouter
from models.seqorth_config import SeqorthTextConfig
from evaluation.callbacks import SpecHornScheduler

def test_router_initialization():
    """Test SeqorthRouter initialization with SpecHorn parameters"""
    print("\n" + "="*80)
    print("TEST 1: Router Initialization")
    print("="*80)
    
    config = SeqorthTextConfig(
        hidden_size=512,
        n_routed_experts=32,
        router_dim=128,
        routed_scaling_factor=1.0,
        spechorn_bias_scale=8.0,
        spechorn_cap_penalty_scale=15.0,
        spechorn_ortho_scale=0.4,
        spechorn_sinkhorn_eps=0.05,
        spechorn_sinkhorn_iter=4,
        spechorn_spec_update_every=16,
    )
    
    router = SeqorthRouter(config)
    
    # Check that SpecHorn parameters are set correctly
    assert hasattr(router, 'bias_scale'), "Missing bias_scale"
    assert hasattr(router, 'cap_penalty_scale'), "Missing cap_penalty_scale"
    assert hasattr(router, 'ortho_scale'), "Missing ortho_scale"
    assert hasattr(router, 'sinkhorn_eps'), "Missing sinkhorn_eps"
    assert hasattr(router, 'sinkhorn_iter'), "Missing sinkhorn_iter"
    assert hasattr(router, 'spec_update_every'), "Missing spec_update_every"
    
    # Check EMA buffers
    assert hasattr(router, 'load_ema'), "Missing load_ema buffer"
    assert hasattr(router, 'spec_vec'), "Missing spec_vec buffer"
    assert hasattr(router, 'spec_ema'), "Missing spec_ema buffer"
    assert hasattr(router, 'batch_capacity_used'), "Missing batch_capacity_used buffer"
    assert hasattr(router, 'global_step'), "Missing global_step buffer"
    
    # Check that old components are removed
    assert not hasattr(router, 'expression_projector'), "expression_projector should be removed"
    assert not hasattr(router, 'bias_predictor'), "bias_predictor should be removed"
    assert not hasattr(router, 'contrastive_loss'), "contrastive_loss should be removed"
    
    print("✅ Router initialization test passed!")
    print(f"  - bias_scale: {router.bias_scale}")
    print(f"  - cap_penalty_scale: {router.cap_penalty_scale}")
    print(f"  - ortho_scale: {router.ortho_scale}")
    print(f"  - sinkhorn_eps: {router.sinkhorn_eps}")
    print(f"  - sinkhorn_iter: {router.sinkhorn_iter}")
    print(f"  - spec_update_every: {router.spec_update_every}")
    print(f"  - load_ema shape: {router.load_ema.shape}")
    print(f"  - spec_ema shape: {router.spec_ema.shape}")
    
    return router


def test_router_forward():
    """Test SeqorthRouter forward pass"""
    print("\n" + "="*80)
    print("TEST 2: Router Forward Pass")
    print("="*80)
    
    config = SeqorthTextConfig(
        hidden_size=512,
        n_routed_experts=32,
        router_dim=128,
        routed_scaling_factor=1.0,
        spechorn_bias_scale=8.0,
        spechorn_cap_penalty_scale=15.0,
        spechorn_ortho_scale=0.4,
        spechorn_sinkhorn_eps=0.05,
        spechorn_sinkhorn_iter=4,
        spechorn_spec_update_every=16,
    )
    
    router = SeqorthRouter(config)
    router.train()
    
    # Create dummy input
    batch_size = 2
    seq_len = 4
    hidden_size = config.hidden_size
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    hn = torch.randn(1, batch_size, config.n_routed_experts * config.router_dim)
    
    # Forward pass
    try:
        outputs = router(x, hn, top_k=4, jitter_eps=0.01)
        
        # Unpack outputs
        (multiplier, selected_experts, expression_logits, hn_out,
         speciality_loss, domain_orthogonality, contrastive_loss, routing_probs,
         expression_reg_loss, routing_uncertainty, entropy_loss,
         load_balancing_loss, sinkhorn_loss, ortho_loss) = outputs
        
        # Check output shapes
        expected_batch_seq = batch_size * seq_len
        assert multiplier.shape == (expected_batch_seq, 4), f"multiplier shape mismatch: {multiplier.shape}"
        assert selected_experts.shape == (expected_batch_seq, 4), f"selected_experts shape mismatch: {selected_experts.shape}"
        assert expression_logits is None, "expression_logits should be None"
        assert hn_out.shape == hn.shape, f"hn_out shape mismatch: {hn_out.shape}"
        
        # Check that losses are all zero (Loss-Free)
        assert speciality_loss.item() == 0.0, "speciality_loss should be 0"
        assert contrastive_loss.item() == 0.0, "contrastive_loss should be 0"
        assert expression_reg_loss.item() == 0.0, "expression_reg_loss should be 0"
        assert routing_uncertainty.item() == 0.0, "routing_uncertainty should be 0"
        assert entropy_loss.item() == 0.0, "entropy_loss should be 0"
        assert load_balancing_loss.item() == 0.0, "load_balancing_loss should be 0"
        assert sinkhorn_loss.item() == 0.0, "sinkhorn_loss should be 0"
        assert ortho_loss.item() == 0.0, "ortho_loss should be 0"
        
        print("✅ Router forward pass test passed!")
        print(f"  - multiplier shape: {multiplier.shape}")
        print(f"  - selected_experts shape: {selected_experts.shape}")
        print(f"  - routing_probs shape: {routing_probs.shape}")
        print(f"  - All losses are 0.0 (Loss-Free) ✓")
        print(f"  - global_step after forward: {router.global_step.item()}")
        
    except Exception as e:
        print(f"❌ Router forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_spechorn_scheduler():
    """Test SpecHornScheduler callback"""
    print("\n" + "="*80)
    print("TEST 3: SpecHornScheduler Callback")
    print("="*80)
    
    scheduler = SpecHornScheduler(
        target_cv_min=0.03,
        target_cv_max=0.08,
        cap_penalty_min=5.0,
        cap_penalty_max=30.0,
        cap_penalty_step=1.0,
        bias_scale_min=4.0,
        bias_scale_max=12.0,
        ortho_scale_min=0.1,
        ortho_scale_max=0.6,
        use_wandb=False,  # Disable wandb for testing
    )
    
    # Check initialization
    assert scheduler.target_cv_min == 0.03
    assert scheduler.target_cv_max == 0.08
    assert scheduler.cap_penalty_min == 5.0
    assert scheduler.cap_penalty_max == 30.0
    
    print("✅ SpecHornScheduler callback test passed!")
    print(f"  - target_cv_min: {scheduler.target_cv_min}")
    print(f"  - target_cv_max: {scheduler.target_cv_max}")
    print(f"  - cap_penalty range: [{scheduler.cap_penalty_min}, {scheduler.cap_penalty_max}]")
    print(f"  - bias_scale range: [{scheduler.bias_scale_min}, {scheduler.bias_scale_max}]")
    print(f"  - ortho_scale range: [{scheduler.ortho_scale_min}, {scheduler.ortho_scale_max}]")


def test_config_parameters():
    """Test that config file has SpecHorn parameters"""
    print("\n" + "="*80)
    print("TEST 4: Config File Parameters")
    print("="*80)
    
    import json
    config_path = "training/seqorth_sft/config/seqorth_small_config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    moe_params = config['model_config']['seqorth_params']
    
    # Check SpecHorn parameters
    required_params = [
        'routed_scaling_factor',
        'spechorn_bias_scale',
        'spechorn_cap_penalty_scale',
        'spechorn_ortho_scale',
        'spechorn_sinkhorn_eps',
        'spechorn_sinkhorn_iter',
        'spechorn_spec_update_every',
    ]
    
    for param in required_params:
        assert param in moe_params, f"Missing parameter: {param}"
    
    print("✅ Config file test passed!")
    print("  SpecHorn parameters in config:")
    for param in required_params:
        print(f"    - {param}: {moe_params[param]}")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("SPECHORN INTEGRATION TESTS")
    print("="*80)
    
    try:
        # Test 1: Router initialization
        router = test_router_initialization()
        
        # Test 2: Router forward pass
        test_router_forward()
        
        # Test 3: SpecHornScheduler callback
        test_spechorn_scheduler()
        
        # Test 4: Config file parameters
        test_config_parameters()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nSpecHorn integration is complete and working correctly.")
        print("You can now run training with the new SpecHorn routing mechanism.")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TESTS FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
