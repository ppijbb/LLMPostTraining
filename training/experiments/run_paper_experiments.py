#!/usr/bin/env python3
"""
Master Experiment Orchestration Script

Orchestrates all experiments for SPECTRA MoE paper:
1. Baseline training (Dense, Standard MoE)
2. SPECTRA training (full, ablations)
3. Benchmark evaluation on all models
4. Efficiency measurement
5. Expert specialization analysis
6. Result aggregation and reporting
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_training(
    model_path: str,
    config_path: str,
    output_dir: Path,
    variant_name: str,
    use_SPECTRA: bool = True,
    ablation_type: str = "none",
) -> Path:
    """
    Run training for a model variant.
    
    Returns:
        Path to trained model checkpoint
    """
    print(f"\n{'='*60}")
    print(f"Training: {variant_name}")
    print(f"{'='*60}")
    
    checkpoint_dir = output_dir / "checkpoints" / variant_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training config
    with open(config_path, 'r') as f:
        train_config = json.load(f)
    
    # Modify config for variant
    if use_SPECTRA:
        if ablation_type != "none":
            train_config["ablation_type"] = ablation_type
    else:
        train_config["use_standard_moe"] = True
    
    # Save modified config
    variant_config_path = checkpoint_dir / "training_config.json"
    with open(variant_config_path, 'w') as f:
        json.dump(train_config, f, indent=2)
    
    # Run training (this would call the actual training script)
    # For now, return the checkpoint directory
    print(f"Training configuration saved to {variant_config_path}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # In practice, you would run:
    # subprocess.run([
    #     "python", "sft/custom_model_sft.py",
    #     "--config", str(variant_config_path),
    #     "--output_dir", str(checkpoint_dir),
    # ])
    
    return checkpoint_dir


def run_benchmark_evaluation(
    model_path: Path,
    output_dir: Path,
    variant_name: str,
) -> Path:
    """
    Run benchmark evaluation on a model.
    
    Returns:
        Path to evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {variant_name}")
    print(f"{'='*60}")
    
    results_dir = output_dir / "benchmark_results" / variant_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark suite
    cmd = [
        "python", "eval/run_full_benchmark_suite.py",
        "--model_path", str(model_path),
        "--output_dir", str(results_dir),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    # subprocess.run(cmd)  # Uncomment to actually run
    
    return results_dir


def run_efficiency_measurement(
    model_path: Path,
    output_dir: Path,
    variant_name: str,
) -> Path:
    """
    Measure efficiency metrics.
    
    Returns:
        Path to efficiency results
    """
    print(f"\n{'='*60}")
    print(f"Measuring efficiency: {variant_name}")
    print(f"{'='*60}")
    
    results_dir = output_dir / "efficiency_results" / variant_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run efficiency measurement
    cmd = [
        "python", "eval/measure_efficiency.py",
        "--model_path", str(model_path),
        "--output_dir", str(results_dir),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    # subprocess.run(cmd)  # Uncomment to actually run
    
    return results_dir


def run_expert_analysis(
    model_path: Path,
    dataset_path: str,
    output_dir: Path,
    variant_name: str,
) -> Path:
    """
    Analyze expert specialization.
    
    Returns:
        Path to analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing experts: {variant_name}")
    print(f"{'='*60}")
    
    results_dir = output_dir / "expert_analysis" / variant_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run expert analysis
    cmd = [
        "python", "eval/analyze_expert_specialization.py",
        "--model_path", str(model_path),
        "--dataset", dataset_path,
        "--output_dir", str(results_dir),
        "--visualize",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    # subprocess.run(cmd)  # Uncomment to actually run
    
    return results_dir


def main():
    parser = argparse.ArgumentParser(description="Master Experiment Orchestration")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model path or HuggingFace identifier",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        required=True,
        help="Path to training configuration JSON",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset (text file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./paper_experiments",
        help="Output directory for all experiments",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["dense", "standard_moe", "SPECTRA"],
        help="Model variants to train",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=["none", "no_expression", "no_gru", "no_penalty", "no_ortho"],
        help="Ablation variants to test",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training (use existing checkpoints)",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--skip_efficiency",
        action="store_true",
        help="Skip efficiency measurement",
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip expert analysis",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment plan
    experiment_plan = {
        "timestamp": datetime.now().isoformat(),
        "base_model": args.base_model,
        "models": args.models,
        "ablations": args.ablations,
        "results": {},
    }
    
    # Run experiments for each model variant
    for model_variant in args.models:
        print(f"\n{'='*80}")
        print(f"Processing Model Variant: {model_variant}")
        print(f"{'='*80}")
        
        variant_results = {}
        
        # Determine ablation variants
        if model_variant == "SPECTRA":
            variants_to_test = args.ablations
        else:
            variants_to_test = ["none"]  # Only one variant for dense/standard_moe
        
        for ablation in variants_to_test:
            variant_name = f"{model_variant}_{ablation}" if ablation != "none" else model_variant
            
            print(f"\n{'='*60}")
            print(f"Variant: {variant_name}")
            print(f"{'='*60}")
            
            variant_results[variant_name] = {}
            
            # Training
            if not args.skip_training:
                checkpoint_dir = run_training(
                    model_path=args.base_model,
                    config_path=args.training_config,
                    output_dir=output_dir,
                    variant_name=variant_name,
                    use_SPECTRA=(model_variant == "SPECTRA"),
                    ablation_type=ablation,
                )
                variant_results[variant_name]["checkpoint_dir"] = str(checkpoint_dir)
            else:
                # Assume checkpoint exists
                checkpoint_dir = output_dir / "checkpoints" / variant_name
                variant_results[variant_name]["checkpoint_dir"] = str(checkpoint_dir)
            
            # Benchmark evaluation
            if not args.skip_eval:
                benchmark_results = run_benchmark_evaluation(
                    model_path=checkpoint_dir,
                    output_dir=output_dir,
                    variant_name=variant_name,
                )
                variant_results[variant_name]["benchmark_results"] = str(benchmark_results)
            
            # Efficiency measurement
            if not args.skip_efficiency:
                efficiency_results = run_efficiency_measurement(
                    model_path=checkpoint_dir,
                    output_dir=output_dir,
                    variant_name=variant_name,
                )
                variant_results[variant_name]["efficiency_results"] = str(efficiency_results)
            
            # Expert analysis
            if not args.skip_analysis:
                expert_results = run_expert_analysis(
                    model_path=checkpoint_dir,
                    dataset_path=args.eval_dataset,
                    output_dir=output_dir,
                    variant_name=variant_name,
                )
                variant_results[variant_name]["expert_analysis"] = str(expert_results)
        
        experiment_plan["results"][model_variant] = variant_results
    
    # Save experiment plan
    plan_path = output_dir / "experiment_plan.json"
    with open(plan_path, 'w') as f:
        json.dump(experiment_plan, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Experiment Orchestration Complete")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Experiment plan: {plan_path}")
    
    # Generate report
    print("\nGenerating final report...")
    report_cmd = [
        "python", "experiments/generate_paper_report.py",
        "--experiment_dir", str(output_dir),
        "--output_dir", str(output_dir / "paper_report"),
    ]
    print(f"Run: {' '.join(report_cmd)}")


if __name__ == "__main__":
    main()

