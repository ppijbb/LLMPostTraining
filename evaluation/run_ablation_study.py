#!/usr/bin/env python3
"""
Ablation Study Automation for SPECTRA MoE

Runs ablation experiments by training and evaluating different variants:
- Full SPECTRA
- -Expression
- -GRU
- -SpecialityPenalty
- -OrthoConstraint
- StandardRouter
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.spectra_ablation import create_ablation_router
from models.spectra import upcycle_model_to_moe, extract_config_info
from models.standard_moe_upcycle import upcycle_to_switch_moe


def create_ablation_model(
    model,
    moe_config: Dict[str, Any],
    ablation_type: str = "none",
    verbose: bool = True,
):
    """
    Create an ablation variant model.
    
    Args:
        model: Base model
        moe_config: MoE configuration
        ablation_type: Type of ablation (none, no_expression, no_gru, no_penalty, no_ortho, standard_router)
        verbose: Whether to show progress
        
    Returns:
        Model with ablation variant applied
    """
    import copy
    import torch.nn as nn
    
    # Create a copy of the model
    ablation_model = copy.deepcopy(model)
    
    if ablation_type == "standard_router":
        # Use Switch-style router
        from models.standard_moe_upcycle import upcycle_to_switch_moe
        ablation_model = upcycle_to_switch_moe(
            ablation_model,
            moe_config,
            verbose=verbose,
        )
    else:
        # Use SPECTRA with ablation router
        # Modify upcycle_model_to_moe to accept custom router
        # For now, we'll create a modified version
        from models.spectra import (
            find_layers_in_model,
            find_mlp_in_layer,
            is_already_moe,
            copy_mlp_weights_to_expert,
        )
        from models.spectra_ablation import create_ablation_router, create_ablation_moe_block
        from models.g3moe_model import G3MoEMLP
        
        cfg = extract_config_info(moe_config)
        
        # Detect model dtype
        model_dtype = None
        for param in ablation_model.parameters():
            if param is not None:
                model_dtype = param.dtype
                break
        if model_dtype is None:
            model_dtype = torch.float32
        
        # Create ablation router
        router = create_ablation_router(
            hidden_size=cfg['hidden_size'],
            num_experts=cfg['num_experts'],
            router_dim=cfg['router_dim'],
            balancing_strength=cfg['balancing_strength'],
            ema_alpha=cfg['ema_alpha'],
            ablation_type=ablation_type,
        )
        router = router.to(dtype=model_dtype)
        
        # Find layers
        layers = find_layers_in_model(ablation_model)
        if layers is None:
            raise ValueError("Could not find decoder layers in model")
        
        # Create dummy config
        class DummyConfig:
            def __init__(self):
                self.hidden_size = cfg['hidden_size']
                self.intermediate_size = cfg['intermediate_size']
                self.hidden_activation = cfg['hidden_activation']
        
        expert_config = DummyConfig()
        
        # Replace MLP layers
        for layer_idx, decoder_layer in enumerate(layers):
            if layer_idx < cfg['first_k_dense_replace']:
                continue
            
            source_mlp = find_mlp_in_layer(decoder_layer)
            if source_mlp is None:
                continue
            
            if is_already_moe(source_mlp):
                continue  # Skip if already MoE
            
            # Create MoE block with ablation router
            moe_block = create_ablation_moe_block(
                router=router,
                expert_module_class=G3MoEMLP,
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
            
            # Replace MLP
            for attr_name in ['mlp', 'feed_forward', 'ffn', 'ffw']:
                if hasattr(decoder_layer, attr_name):
                    setattr(decoder_layer, attr_name, moe_block)
                    break
            
            # Copy weights
            if hasattr(moe_block, 'shared_experts'):
                copy_mlp_weights_to_expert(source_mlp, moe_block.shared_experts)
            
            if hasattr(moe_block, 'experts'):
                for expert in moe_block.experts:
                    copy_mlp_weights_to_expert(source_mlp, expert)
    
    return ablation_model


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Study for SPECTRA MoE")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base model or HuggingFace model identifier",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["none", "no_expression", "no_gru", "no_penalty", "no_ortho", "standard_router"],
        help="Ablation variants to test",
    )
    parser.add_argument(
        "--moe_config",
        type=str,
        required=True,
        help="Path to MoE configuration JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ablation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate (skip training)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MoE config
    with open(args.moe_config, 'r') as f:
        moe_config = json.load(f)
    
    # Load base model
    print(f"Loading base model from {args.base_model}...")
    from transformers import AutoModelForCausalLM
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    results = {}
    
    # Create and evaluate each variant
    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"Processing variant: {variant}")
        print(f"{'='*60}")
        
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ablation model
        print(f"Creating {variant} model...")
        ablation_model = create_ablation_model(
            base_model,
            moe_config,
            ablation_type=variant,
            verbose=True,
        )
        
        # Save model
        model_path = variant_dir / "model"
        ablation_model.save_pretrained(model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate (if eval_only or if training is done)
        if args.eval_only:
            print(f"Evaluating {variant}...")
            # Run benchmark evaluation
            # This would call run_full_benchmark_suite.py or similar
            # For now, just save the model path
            results[variant] = {
                "model_path": str(model_path),
                "status": "created",
            }
    
    # Save results
    results_path = output_dir / "ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Ablation Study Complete")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    import torch
    main()

