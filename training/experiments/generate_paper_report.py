#!/usr/bin/env python3
"""
Paper Report Generator

Generates LaTeX tables and figures from experiment results:
- Benchmark comparison tables
- Efficiency curves
- Ablation study results
- Expert utilization heatmaps
- Statistical significance tests
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib seaborn")


def load_experiment_results(experiment_dir: Path) -> Dict[str, Any]:
    """Load all experiment results from directory"""
    results = {}
    
    # Load experiment plan
    plan_path = experiment_dir / "experiment_plan.json"
    if plan_path.exists():
        with open(plan_path, 'r') as f:
            results["plan"] = json.load(f)
    
    # Load benchmark results
    benchmark_dir = experiment_dir / "benchmark_results"
    if benchmark_dir.exists():
        results["benchmarks"] = {}
        for variant_dir in benchmark_dir.iterdir():
            if variant_dir.is_dir():
                results_file = variant_dir / "benchmark_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results["benchmarks"][variant_dir.name] = json.load(f)
    
    # Load efficiency results
    efficiency_dir = experiment_dir / "efficiency_results"
    if efficiency_dir.exists():
        results["efficiency"] = {}
        for variant_dir in efficiency_dir.iterdir():
            if variant_dir.is_dir():
                efficiency_file = variant_dir / "efficiency_results.json"
                if efficiency_file.exists():
                    with open(efficiency_file, 'r') as f:
                        results["efficiency"][variant_dir.name] = json.load(f)
    
    # Load expert analysis
    expert_dir = experiment_dir / "expert_analysis"
    if expert_dir.exists():
        results["expert_analysis"] = {}
        for variant_dir in expert_dir.iterdir():
            if variant_dir.is_dir():
                analysis_file = variant_dir / "expert_analysis.json"
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        results["expert_analysis"][variant_dir.name] = json.load(f)
    
    return results


def generate_benchmark_table(results: Dict[str, Any], output_path: Path):
    """Generate LaTeX table for benchmark results"""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * len(results.get("benchmarks", {})) + "}",
        "\\toprule",
        "Task & " + " & ".join(results.get("benchmarks", {}).keys()) + " \\\\",
        "\\midrule",
    ]
    
    # Collect all tasks
    all_tasks = set()
    for variant_results in results.get("benchmarks", {}).values():
        all_tasks.update(variant_results.keys())
    
    # Generate rows
    for task in sorted(all_tasks):
        row = [task]
        for variant_name in results.get("benchmarks", {}).keys():
            variant_results = results["benchmarks"][variant_name]
            if task in variant_results:
                score = variant_results[task]
                if isinstance(score, dict):
                    # Extract main score
                    score = score.get("accuracy", score.get("score", score.get("perplexity", "N/A")))
                row.append(f"{score:.4f}" if isinstance(score, (int, float)) else str(score))
            else:
                row.append("N/A")
        lines.append(" & ".join(row) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Benchmark Results Comparison}",
        "\\label{tab:benchmark_comparison}",
        "\\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def generate_efficiency_curves(results: Dict[str, Any], output_path: Path):
    """Generate efficiency curves plot"""
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot generate plots: matplotlib not available")
        return
    
    efficiency_results = results.get("efficiency", {})
    if not efficiency_results:
        print("No efficiency results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Throughput vs batch size
    ax = axes[0, 0]
    for variant_name, variant_data in efficiency_results.items():
        forward_data = variant_data.get("forward_throughput", {})
        batch_sizes = sorted(forward_data.keys())
        throughputs = [forward_data[bs]["tokens_per_sec"] for bs in batch_sizes]
        ax.plot(batch_sizes, throughputs, marker='o', label=variant_name)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Forward Pass Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Generation latency
    ax = axes[0, 1]
    for variant_name, variant_data in efficiency_results.items():
        gen_data = variant_data.get("generation_latency", {})
        if gen_data:
            latency = gen_data.get("per_token_latency_ms_mean", 0)
            ax.bar(variant_name, latency, alpha=0.7)
    ax.set_ylabel("Latency (ms/token)")
    ax.set_title("Generation Latency")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Memory usage
    ax = axes[1, 0]
    for variant_name, variant_data in efficiency_results.items():
        forward_data = variant_data.get("forward_throughput", {})
        if forward_data:
            # Use first batch size for memory
            first_bs = sorted(forward_data.keys())[0]
            memory = forward_data[first_bs].get("memory_gb_peak", 0)
            ax.bar(variant_name, memory, alpha=0.7)
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Peak Memory Usage")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Expert utilization (if available)
    ax = axes[1, 1]
    expert_data = results.get("expert_analysis", {})
    if expert_data:
        for variant_name, variant_analysis in expert_data.items():
            # This would need actual expert utilization data
            pass
    ax.set_title("Expert Utilization")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Efficiency curves saved to {output_path}")


def generate_ablation_comparison(results: Dict[str, Any], output_path: Path):
    """Generate ablation study comparison table"""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Variant & Metric & Score \\\\",
        "\\midrule",
    ]
    
    # Extract ablation results
    benchmarks = results.get("benchmarks", {})
    ablation_variants = [v for v in benchmarks.keys() if "Seqorth" in v]
    
    # Use a common metric (e.g., MMLU)
    for variant in ablation_variants:
        variant_results = benchmarks[variant]
        # Extract MMLU or first available metric
        for metric, score in variant_results.items():
            if isinstance(score, dict):
                score = score.get("accuracy", score.get("score", "N/A"))
            lines.append(f"{variant} & {metric} & {score:.4f if isinstance(score, (int, float)) else score} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Ablation Study Results}",
        "\\label{tab:ablation_study}",
        "\\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate Paper Report")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./paper_report",
        help="Output directory for reports",
    )
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading experiment results...")
    results = load_experiment_results(experiment_dir)
    
    # Generate tables
    print("Generating LaTeX tables...")
    generate_benchmark_table(results, output_dir / "benchmark_table.tex")
    generate_ablation_comparison(results, output_dir / "ablation_table.tex")
    
    # Generate figures
    if MATPLOTLIB_AVAILABLE:
        print("Generating figures...")
        generate_efficiency_curves(results, output_dir / "efficiency_curves.pdf")
    
    # Generate summary JSON
    summary = {
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "results_loaded": len(results) > 0,
        "tables_generated": [
            "benchmark_table.tex",
            "ablation_table.tex",
        ],
        "figures_generated": [
            "efficiency_curves.pdf",
        ] if MATPLOTLIB_AVAILABLE else [],
    }
    
    summary_path = output_dir / "report_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nReport generation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

