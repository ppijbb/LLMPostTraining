#!/usr/bin/env python3
"""
D+3~4: Expert Analysis - SPECTRA's Core Novelty

Analyze and visualize expert specialization:
1. Domain-specific expert usage heatmaps
2. GRU trajectory consistency analysis
3. Representation orthogonality (OSR validation)
"""

import os
import sys
import argparse
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.routing_benchmarks.utils import (
    load_spectra_checkpoint,
    prepare_model_for_eval,
    MetricTracker,
    plot_expert_heatmap,
    plot_trajectory_consistency,
    plot_orthogonality_histogram
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_expert_usage(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    layer_range: Tuple[int, int],
    batch_size: int = 4
) -> Dict[int, np.ndarray]:
    """
    Collect expert usage statistics across layers.
    
    Args:
        model: Model to analyze
        tokenizer: Tokenizer
        texts: List of text samples
        layer_range: Tuple of (start_layer, end_layer)
        batch_size: Batch size for processing
    
    Returns:
        Dictionary mapping layer_idx -> expert_usage_array
    """
    model.eval()
    
    # Get number of experts from model
    num_experts = getattr(model.config, 'num_experts', 256)
    
    # Storage for expert usage counts
    expert_usage = defaultdict(lambda: np.zeros(num_experts))
    
    # Hook to capture routing decisions
    routing_decisions = {}
    
    def routing_hook(name, layer_idx):
        def hook(module, input, output):
            # Assuming output contains router_logits or expert_indices
            if isinstance(output, tuple):
                router_output = output[1] if len(output) > 1 else output[0]
            else:
                router_output = output
            
            # Extract expert indices (top-k selection)
            if hasattr(router_output, 'expert_indices'):
                expert_indices = router_output.expert_indices
            elif isinstance(router_output, dict) and 'expert_indices' in router_output:
                expert_indices = router_output['expert_indices']
            else:
                # Try to get from router_logits
                if hasattr(router_output, 'router_logits'):
                    router_logits = router_output.router_logits
                    expert_indices = router_logits.topk(
                        getattr(model.config, 'num_experts_per_tok', 8), 
                        dim=-1
                    )[1]
                else:
                    return
            
            routing_decisions[layer_idx] = expert_indices.detach().cpu()
        
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in range(*layer_range):
        try:
            # Try different possible MoE module paths
            layer_module = None
            if hasattr(model, 'model'):
                if hasattr(model.model, 'layers'):
                    layer = model.model.layers[layer_idx]
                elif hasattr(model.model, 'decoder'):
                    layer = model.model.decoder.layers[layer_idx]
            
            # Find MoE module in layer
            if hasattr(layer, 'moe'):
                layer_module = layer.moe
            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                layer_module = layer.mlp
            
            if layer_module:
                hook = layer_module.register_forward_hook(
                    routing_hook(f"layer_{layer_idx}", layer_idx)
                )
                hooks.append(hook)
        except Exception as e:
            logger.warning(f"Could not register hook for layer {layer_idx}: {e}")
    
    # Process texts
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                # Forward pass (hooks will capture routing)
                routing_decisions.clear()
                outputs = model(**inputs)
                
                # Accumulate expert usage
                for layer_idx, expert_indices in routing_decisions.items():
                    # expert_indices shape: [batch, seq_len, top_k]
                    flat_indices = expert_indices.flatten().numpy()
                    unique, counts = np.unique(flat_indices, return_counts=True)
                    expert_usage[layer_idx][unique] += counts
                    
            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return dict(expert_usage)


def compute_trajectory_consistency(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    num_layers: int = 32
) -> np.ndarray:
    """
    Compute GRU trajectory consistency (L1 distance between layers).
    
    Args:
        model: Model to analyze
        tokenizer: Tokenizer
        texts: List of text samples
        num_layers: Number of layers
    
    Returns:
        Array of consistency scores (lower = more consistent)
    """
    model.eval()
    
    consistency_scores = []
    
    # Hook to capture router logits
    router_logits_per_layer = {}
    
    def logits_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                router_output = output[1] if len(output) > 1 else output[0]
            else:
                router_output = output
            
            if hasattr(router_output, 'router_logits'):
                router_logits_per_layer[layer_idx] = router_output.router_logits.detach().cpu()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in range(num_layers):
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layer = model.model.layers[layer_idx]
                if hasattr(layer, 'moe'):
                    hook = layer.moe.register_forward_hook(logits_hook(layer_idx))
                    hooks.append(hook)
        except:
            pass
    
    # Process texts
    with torch.no_grad():
        for text in texts:
            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(model.device)
                
                router_logits_per_layer.clear()
                outputs = model(**inputs)
                
                # Compute L1 distance between consecutive layers
                layer_indices = sorted(router_logits_per_layer.keys())
                for i in range(len(layer_indices) - 1):
                    logits1 = router_logits_per_layer[layer_indices[i]]
                    logits2 = router_logits_per_layer[layer_indices[i+1]]
                    
                    # Normalize logits to probabilities
                    probs1 = torch.softmax(logits1, dim=-1)
                    probs2 = torch.softmax(logits2, dim=-1)
                    
                    # L1 distance
                    l1_dist = torch.abs(probs1 - probs2).mean().item()
                    consistency_scores.append(l1_dist)
                    
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return np.array(consistency_scores)


def extract_representation_orthogonality(
    model: torch.nn.Module
) -> np.ndarray:
    """
    Extract pairwise cosine similarities of expert representations.
    
    Args:
        model: Model to analyze
    
    Returns:
        Array of pairwise cosine similarities
    """
    # Try to find ExpressionProjector weights
    projector_weights = []
    
    for name, param in model.named_parameters():
        if 'expression_projector' in name.lower() or 'expert_proj' in name.lower():
            if 'weight' in name:
                projector_weights.append(param.detach().cpu())
    
    if not projector_weights:
        logger.warning("Could not find ExpressionProjector weights")
        return np.array([])
    
    # Compute pairwise cosine similarities
    all_similarities = []
    
    for weight_matrix in projector_weights:
        # weight_matrix shape: [num_experts, hidden_dim] or [hidden_dim, num_experts]
        if weight_matrix.dim() == 2:
            # Normalize each expert vector
            norms = torch.norm(weight_matrix, dim=-1, keepdim=True)
            normalized = weight_matrix / (norms + 1e-8)
            
            # Pairwise cosine similarity
            similarity_matrix = torch.mm(normalized, normalized.t())
            
            # Extract upper triangular (excluding diagonal)
            triu_indices = torch.triu_indices(
                similarity_matrix.size(0),
                similarity_matrix.size(1),
                offset=1
            )
            similarities = similarity_matrix[triu_indices[0], triu_indices[1]].numpy()
            all_similarities.extend(similarities)
    
    return np.array(all_similarities)


def main():
    parser = argparse.ArgumentParser(description="D+3~4: Expert Analysis")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation_config.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results/day3_4",
        help="Output directory"
    )
    parser.add_argument(
        "--skip_heatmaps",
        action="store_true",
        help="Skip domain heatmap generation"
    )
    parser.add_argument(
        "--skip_trajectory",
        action="store_true",
        help="Skip trajectory consistency analysis"
    )
    parser.add_argument(
        "--skip_orthogonality",
        action="store_true",
        help="Skip orthogonality analysis"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metric tracker
    tracker = MetricTracker(
        base_dir=config["output"]["base_dir"],
        use_timestamp=False
    )
    
    # Load model
    checkpoint_path = args.checkpoint or config["model"]["checkpoint_path"]
    logger.info(f"Loading model from {checkpoint_path}")
    
    model, tokenizer = load_spectra_checkpoint(
        checkpoint_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = prepare_model_for_eval(model)
    
    # ===== Step 1: Domain-Specific Expert Usage =====
    if not args.skip_heatmaps:
        logger.info("=" * 80)
        logger.info("Analyzing domain-specific expert usage")
        logger.info("=" * 80)
        
        domains_config = config["day3_4_expert_analysis"]["domains"]
        
        # Load domain-specific samples (simplified - you'd load real datasets)
        from datasets import load_dataset
        
        domain_texts = {}
        
        for domain_name, domain_config in domains_config.items():
            logger.info(f"Loading {domain_name} samples...")
            num_samples = domain_config["num_samples"]
            
            # Load appropriate dataset based on domain
            # This is simplified - actual implementation would need proper dataset loading
            try:
                if domain_name == "arxiv":
                    dataset = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
                    texts = [item["article"][:1000] for item in dataset.take(num_samples)]
                elif domain_name == "github":
                    dataset = load_dataset("codeparrot/github-code", split="train", streaming=True)
                    texts = [item["code"][:1000] for item in dataset.take(num_samples)]
                elif domain_name == "novels":
                    dataset = load_dataset("bookcorpus", split="train", streaming=True)
                    texts = [item["text"][:1000] for item in dataset.take(num_samples)]
                else:
                    texts = [f"Sample text {i} for {domain_name}" for i in range(num_samples)]
                
                domain_texts[domain_name] = texts
                logger.info(f"Loaded {len(texts)} samples for {domain_name}")
                
            except Exception as e:
                logger.warning(f"Could not load real data for {domain_name}: {e}")
                domain_texts[domain_name] = [f"Sample {i}" for i in range(num_samples)]
        
        # Collect expert usage for each domain
        layer_range = tuple(config["day3_4_expert_analysis"]["routing_analysis"]["layer_range"])
        
        domain_expert_usage = {}
        
        for domain_name, texts in domain_texts.items():
            logger.info(f"Collecting expert usage for {domain_name}...")
            usage_by_layer = collect_expert_usage(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                layer_range=layer_range
            )
            
            # Average across layers
            all_usage = np.stack(list(usage_by_layer.values()))
            avg_usage = all_usage.mean(axis=0)
            domain_expert_usage[domain_name] = avg_usage
            
            tracker.add_metric("day3_4", f"expert_usage_{domain_name}", avg_usage.tolist())
        
        # Generate heatmap
        logger.info("Generating domain expert usage heatmap...")
        plot_expert_heatmap(
            expert_usage=domain_expert_usage,
            domains=list(domain_expert_usage.keys()),
            output_file=output_dir / "domain_heatmaps.png",
            top_k=config["day3_4_expert_analysis"]["routing_analysis"]["top_k_experts"]
        )
    
    # ===== Step 2: GRU Trajectory Consistency =====
    if not args.skip_trajectory:
        logger.info("=" * 80)
        logger.info("Analyzing GRU trajectory consistency")
        logger.info("=" * 80)
        
        trajectory_config = config["day3_4_expert_analysis"]["trajectory_analysis"]
        
        # Load test samples
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
        texts = [item["text"] for item in dataset if item["text"].strip()][:trajectory_config["num_sequences"]]
        
        logger.info(f"Computing trajectory consistency for {len(texts)} sequences...")
        spectra_consistency = compute_trajectory_consistency(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            num_layers=config["model"]["num_layers"]
        )
        
        tracker.add_metric("day3_4", "trajectory_consistency", {
            "mean": float(spectra_consistency.mean()),
            "std": float(spectra_consistency.std()),
            "scores": spectra_consistency.tolist()
        })
        
        # Plot
        logger.info("Generating trajectory consistency plot...")
        plot_trajectory_consistency(
            spectra_consistency=spectra_consistency,
            output_file=output_dir / "trajectory_consistency.png"
        )
    
    # ===== Step 3: Representation Orthogonality =====
    if not args.skip_orthogonality:
        logger.info("=" * 80)
        logger.info("Analyzing representation orthogonality (OSR)")
        logger.info("=" * 80)
        
        logger.info("Extracting pairwise cosine similarities...")
        similarities = extract_representation_orthogonality(model)
        
        if len(similarities) > 0:
            tracker.add_metric("day3_4", "orthogonality", {
                "mean": float(similarities.mean()),
                "std": float(similarities.std()),
                "near_zero_ratio": float((np.abs(similarities) < 0.1).mean()),
                "similarities": similarities.tolist()
            })
            
            # Plot
            logger.info("Generating orthogonality histogram...")
            plot_orthogonality_histogram(
                similarities=similarities,
                output_file=output_dir / "orthogonality.png"
            )
        else:
            logger.warning("No orthogonality data extracted")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Save results
    tracker.save_day("day3_4")
    tracker.mark_step_complete("day3_4")
    
    logger.info(f"D+3~4 expert analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

