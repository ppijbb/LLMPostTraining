#!/usr/bin/env python3
"""
Expert Specialization Analysis Tool

Analyzes expert activation patterns, specialization, and task correlation:
- Per-expert activation patterns
- t-SNE/PCA visualization
- Expert-task correlation
- Expert output diversity
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not installed. Install with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib seaborn")


def collect_expert_activations(
    model,
    tokenizer,
    dataset: List[str],
    device: str = "cuda",
    max_samples: int = 1000,
) -> Dict[int, List[torch.Tensor]]:
    """
    Collect expert activations for each token.
    
    Returns:
        Dictionary mapping expert_idx -> list of activation tensors
    """
    model.eval()
    expert_activations = defaultdict(list)
    
    # Hook to capture expert outputs
    def create_expert_hook(expert_idx):
        def hook(module, input, output):
            expert_activations[expert_idx].append(output.detach().cpu())
        return hook
    
    # Register hooks on all expert layers
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, 'experts') and isinstance(module.experts, torch.nn.ModuleList):
            for expert_idx, expert in enumerate(module.experts):
                hook = expert.register_forward_hook(create_expert_hook(expert_idx))
                hooks.append(hook)
    
    # Process dataset
    with torch.no_grad():
        for text in tqdm(dataset[:max_samples], desc="Collecting activations"):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return dict(expert_activations)


def compute_expert_similarity(
    expert_activations: Dict[int, List[torch.Tensor]],
) -> np.ndarray:
    """
    Compute pairwise similarity matrix between experts.
    
    Returns:
        Similarity matrix [num_experts, num_experts]
    """
    num_experts = len(expert_activations)
    if num_experts == 0:
        return np.array([])
    
    # Average activations per expert
    expert_means = []
    for expert_idx in sorted(expert_activations.keys()):
        activations = expert_activations[expert_idx]
        if len(activations) > 0:
            # Flatten and average
            flattened = torch.cat([a.flatten() for a in activations])
            expert_means.append(flattened.mean().item())
        else:
            expert_means.append(0.0)
    
    expert_means = np.array(expert_means)
    
    # Compute pairwise cosine similarity
    # For simplicity, use Euclidean distance normalized
    similarity_matrix = np.zeros((num_experts, num_experts))
    for i in range(num_experts):
        for j in range(num_experts):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Simple similarity based on mean activations
                # In practice, you'd want to use actual activation vectors
                similarity_matrix[i, j] = 1.0 - abs(expert_means[i] - expert_means[j]) / (abs(expert_means[i]) + abs(expert_means[j]) + 1e-8)
    
    return similarity_matrix


def visualize_expert_activations(
    expert_activations: Dict[int, List[torch.Tensor]],
    output_path: Path,
    method: str = "tsne",
):
    """
    Visualize expert activation patterns using t-SNE or PCA.
    """
    if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        print("Cannot visualize: sklearn or matplotlib not available")
        return
    
    # Collect all activation vectors
    all_activations = []
    expert_labels = []
    
    for expert_idx in sorted(expert_activations.keys()):
        activations = expert_activations[expert_idx]
        for activation in activations[:100]:  # Limit for visualization
            flattened = activation.flatten().numpy()
            all_activations.append(flattened)
            expert_labels.append(expert_idx)
    
    if len(all_activations) == 0:
        print("No activations to visualize")
        return
    
    # Stack into matrix
    X = np.vstack(all_activations)
    
    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)
    
    X_reduced = reducer.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=expert_labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter, label='Expert Index')
    plt.title(f'Expert Activation Patterns ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visualization saved to {output_path}")


def analyze_expert_task_correlation(
    model,
    tokenizer,
    task_dataset: Dict[str, List[str]],
    device: str = "cuda",
) -> Dict[str, Dict[int, float]]:
    """
    Analyze which experts are activated for which tasks.
    
    Returns:
        Dictionary mapping task_name -> {expert_idx: activation_frequency}
    """
    model.eval()
    task_expert_counts = defaultdict(lambda: defaultdict(int))
    
    # Hook to capture expert selections
    expert_selections = []
    
    def create_routing_hook():
        def hook(module, input, output):
            # Extract selected experts from output
            # This depends on the MoE block structure
            if isinstance(output, tuple) and len(output) > 1:
                # Assume second element contains routing info
                routing_info = output[1]
                if isinstance(routing_info, dict):
                    if 'selected_experts' in routing_info:
                        expert_selections.append(routing_info['selected_experts'].detach().cpu())
        return hook
    
    # Register hooks on MoE blocks
    hooks = []
    for name, module in model.named_modules():
        if 'moe' in name.lower() or 'expert' in name.lower():
            hook = module.register_forward_hook(create_routing_hook())
            hooks.append(hook)
    
    # Process each task
    for task_name, samples in task_dataset.items():
        expert_selections = []
        with torch.no_grad():
            for text in tqdm(samples, desc=f"Processing {task_name}"):
                try:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    _ = model(**inputs)
                except Exception:
                    continue
        
        # Count expert activations per task
        for selection in expert_selections:
            if selection is not None:
                unique_experts = torch.unique(selection)
                for expert_idx in unique_experts:
                    task_expert_counts[task_name][expert_idx.item()] += 1
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Normalize frequencies
    result = {}
    for task_name, counts in task_expert_counts.items():
        total = sum(counts.values())
        if total > 0:
            result[task_name] = {expert_idx: count / total for expert_idx, count in counts.items()}
        else:
            result[task_name] = {}
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze Expert Specialization")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to text dataset file (one text per line)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./expert_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load dataset
    with open(args.dataset, 'r') as f:
        dataset = [line.strip() for line in f if line.strip()]
    
    # Collect expert activations
    print("\nCollecting expert activations...")
    expert_activations = collect_expert_activations(
        model, tokenizer, dataset, device=args.device, max_samples=args.max_samples
    )
    
    # Compute similarity
    print("\nComputing expert similarity...")
    similarity_matrix = compute_expert_similarity(expert_activations)
    
    # Save results
    results = {
        "expert_activations": {
            str(k): len(v) for k, v in expert_activations.items()
        },
        "similarity_matrix": similarity_matrix.tolist() if len(similarity_matrix) > 0 else [],
    }
    
    results_path = output_dir / "expert_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize
    if args.visualize and MATPLOTLIB_AVAILABLE:
        print("\nGenerating visualizations...")
        visualize_expert_activations(
            expert_activations,
            output_dir / "expert_activations_tsne.png",
            method="tsne",
        )
        visualize_expert_activations(
            expert_activations,
            output_dir / "expert_activations_pca.png",
            method="pca",
        )
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

