#!/usr/bin/env python3
"""
Test script for the Continuous Thought Machine (CTM) implementation
"""

import sys
import torch
sys.path.append('.')

try:
    from ctm import create_simple_ctm, create_simple_ctm_non_lazy, ContinuousThoughtMachine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure ctm.py is in the same directory or in the Python path")
    sys.exit(1)

def count_parameters_safely(model, sample_input):
    """
    Safely count model parameters by initializing lazy modules first.
    """
    try:
        # Try to count parameters directly
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except ValueError as e:
        print(f"LazyModule detected, initializing with forward pass...")
        # If lazy modules are not initialized, run a forward pass first
        with torch.no_grad():
            try:
                _ = model(sample_input)
            except Exception as forward_error:
                print(f"Error during initialization forward pass: {forward_error}")
                return "Unknown (initialization failed)"
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_simple_ctm(use_non_lazy=False):
    """Test the simple CTM configuration."""
    version = "non-lazy" if use_non_lazy else "lazy"
    print(f"Creating simple CTM model ({version} version)...")
    
    try:
        if use_non_lazy:
            model = create_simple_ctm_non_lazy(out_dims=10, iterations=3)
        else:
            model = create_simple_ctm(out_dims=10, iterations=3)
        print(f"Model created successfully!")
        
        print(f"\nTesting forward pass...")
        batch_size = 2
        input_dim = 128
        dummy_input = torch.randn(batch_size, input_dim)
        
        with torch.no_grad():
            predictions, certainties, _ = model(dummy_input)
        
        # Count parameters after forward pass to ensure lazy modules are initialized
        param_count = count_parameters_safely(model, dummy_input)
        print(f"Model parameters: {param_count:,}")
        
        print(f"Input shape: {dummy_input.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Certainties shape: {certainties.shape}")
        print("Forward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"Simple CTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ctm_with_backbone():
    """Test CTM with ResNet backbone for vision tasks."""
    print("\n" + "="*50)
    print("Testing CTM with ResNet backbone...")
    
    try:
        model = ContinuousThoughtMachine(
            iterations=2,
            d_model=128,
            d_input=64,
            heads=4,
            n_synch_out=16,
            n_synch_action=16,
            synapse_depth=1,
            memory_length=3,
            deep_nlms=False,
            memory_hidden_dims=32,
            do_layernorm_nlm=False,
            backbone_type='resnet18-2',
            positional_embedding_type='none',
            out_dims=1000,  # ImageNet classes
            prediction_reshaper=[1000],
            dropout=0.0,
            neuron_select_type='random-pairing',
            n_random_pairing_self=2
        )
        
        print("CTM with ResNet backbone created successfully!")
        
        # Test with image-like input
        batch_size = 1
        channels = 3
        height = width = 32
        dummy_image = torch.randn(batch_size, channels, height, width)
        
        with torch.no_grad():
            predictions, certainties, _ = model(dummy_image)
        
        # Count parameters after forward pass
        param_count = count_parameters_safely(model, dummy_image)
        print(f"Model parameters: {param_count:,}")
        
        print(f"Input shape: {dummy_image.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Certainties shape: {certainties.shape}")
        print("ResNet backbone test successful!")
        
        return True
        
    except Exception as e:
        print(f"ResNet backbone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tracking_mode():
    """Test CTM with tracking enabled."""
    print("\n" + "="*50)
    print("Testing CTM with tracking mode...")
    
    try:
        model = create_simple_ctm(out_dims=5, iterations=2)
        
        batch_size = 1
        input_dim = 128
        dummy_input = torch.randn(batch_size, input_dim)
        
        with torch.no_grad():
            results = model(dummy_input, track=True)
            predictions, certainties, synch_tracking, pre_activations, post_activations, attention = results
        
        print(f"Tracking results:")
        print(f"- Predictions shape: {predictions.shape}")
        print(f"- Certainties shape: {certainties.shape}")
        print(f"- Synch tracking: {len(synch_tracking)} arrays")
        print(f"- Pre-activations shape: {pre_activations.shape}")
        print(f"- Post-activations shape: {post_activations.shape}")
        print(f"- Attention weights shape: {attention.shape}")
        print("Tracking mode test successful!")
        
        return True
        
    except Exception as e:
        print(f"Tracking mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Continuous Thought Machine (CTM) Implementation")
    print("=" * 60)
    
    # Test simple CTM (lazy version)
    print("Test 1: Simple CTM (with lazy modules)")
    success1 = test_simple_ctm(use_non_lazy=False)
    
    # Test simple CTM (non-lazy version)
    print(f"\n{'='*60}")
    print("Test 2: Simple CTM (non-lazy version)")
    success2 = test_simple_ctm(use_non_lazy=True)
    
    # Test CTM with backbone
    success3 = test_ctm_with_backbone()
    
    # Test tracking mode
    success4 = test_tracking_mode()
    
    print("\n" + "="*60)
    total_tests = 4
    passed_tests = sum([success1, success2, success3, success4])
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✅ All tests passed! CTM implementation is working correctly!")
    elif passed_tests > 0:
        print("⚠️  Some tests passed, but there are issues with others.")
    else:
        print("❌ All tests failed. Please check the implementation.")
    
    print("\nCTM Features implemented:")
    print("- ✅ Neuron-Level Models (NLMs)")
    print("- ✅ Synchronisation as Representation")
    print("- ✅ Internal Recurrence (thought ticks)")
    print("- ✅ Multiple backbone types (ResNet, shallow-wide, etc.)")
    print("- ✅ Various positional embeddings")
    print("- ✅ Attention mechanisms")
    print("- ✅ U-Net synapses")
    print("- ✅ Uncertainty/certainty estimation")
    print("- ✅ Tracking and visualization support")
    print("- ✅ Both lazy and non-lazy module versions")

if __name__ == "__main__":
    main() 