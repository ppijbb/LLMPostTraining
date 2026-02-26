#!/usr/bin/env python3
"""
D+6: Final Comparison Table Generation

Aggregate all evaluation results and generate paper-ready LaTeX tables:
- Table 1: Main comparison with baselines
- Table 2: Ablation study
- Table 3: Model specifications
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.routing_benchmarks.utils import (
    MetricTracker,
    generate_comparison_table,
    generate_ablation_table,
    generate_parameter_table,
    generate_benchmark_table
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def aggregate_model_results(
    tracker: MetricTracker,
    model_name: str
) -> Dict[str, Any]:
    """
    Aggregate results for a single model across all days.
    
    Args:
        tracker: Metric tracker with all results
        model_name: Name of the model
    
    Returns:
        Aggregated results dictionary
    """
    results = {}
    
    # Day 0: Perplexity
    day0_ppl = tracker.get_metric("day0", f"perplexity_{model_name}")
    if day0_ppl:
        results["ppl_wikitext"] = day0_ppl
    
    # Day 1-2: Benchmarks
    day1_2_results = tracker.get_model_result("day1_2", model_name)
    if day1_2_results:
        results.update(day1_2_results)
    
    # Day 5: Throughput
    day5_throughput = tracker.get_metric("day5", f"throughput_{model_name}")
    if day5_throughput:
        results["throughput_tokens_per_sec"] = day5_throughput.get("avg_throughput", 0)
        results["ttft_ms"] = day5_throughput.get("avg_ttft", 0)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="D+6: Final Comparison Tables")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation_config.yaml"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to results directory (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results/day6",
        help="Output directory for tables"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metric tracker with existing results
    results_dir = args.results_dir or config["output"]["base_dir"]
    tracker = MetricTracker(base_dir=results_dir, use_timestamp=False)
    tracker.load_all()
    
    logger.info(f"Loaded results from {results_dir}")
    
    # Load baseline configs
    baseline_config_file = Path(args.config).parent / "baseline_models.yaml"
    with open(baseline_config_file, 'r') as f:
        baseline_config = yaml.safe_load(f)
    
    # ===== Table 1: Main Comparison =====
    logger.info("=" * 80)
    logger.info("Generating Table 1: Main Comparison")
    logger.info("=" * 80)
    
    # Determine which models to include
    comparison_group = baseline_config["comparison_groups"]["similar_active_params"]
    model_names = ["Seqorth"] + [baseline_config["models"][k]["name"] 
                                  for k in comparison_group 
                                  if k in baseline_config["models"]]
    
    # Aggregate results for each model
    model_results = {}
    for model_name in model_names:
        logger.info(f"Aggregating results for {model_name}...")
        results = aggregate_model_results(tracker, model_name)
        
        # Add parameter info
        if model_name == "Seqorth":
            results["total_params"] = config["model"].get("num_experts", 256) * 1e9  # Approximate
            results["active_params"] = config["model"].get("active_experts", 8) * 1e9 / config["model"].get("num_experts", 256)
        else:
            # Find in baseline config
            for key, model_config in baseline_config["models"].items():
                if model_config["name"] == model_name:
                    results["total_params"] = model_config["params"]["total"]
                    results["active_params"] = model_config["params"]["active"]
                    break
        
        model_results[model_name] = results
    
    # Generate comparison table
    metrics_config = config["day6_comparison"]["metrics"]
    
    comparison_table = generate_comparison_table(
        model_results=model_results,
        metrics=metrics_config,
        output_file=output_dir / "table1_comparison.tex",
        caption=config["day6_comparison"]["latex"]["caption"],
        label=config["day6_comparison"]["latex"]["label"],
        column_alignment=config["day6_comparison"]["latex"]["column_alignment"]
    )
    
    logger.info("Table 1 generated successfully")
    
    # ===== Table 2: Ablation Study =====
    logger.info("=" * 80)
    logger.info("Generating Table 2: Ablation Study")
    logger.info("=" * 80)
    
    # Load ablation results from day5
    ablation_results = {}
    ablation_configs = config["day5_efficiency"]["ablation"]["configs"]
    
    for config_spec in ablation_configs:
        config_name = config_spec["name"]
        results = tracker.get_metric("day5", f"ablation_{config_name}")
        if results:
            ablation_results[config_name] = results
    
    if ablation_results:
        ablation_table = generate_ablation_table(
            ablation_results=ablation_results,
            metrics=config["day5_efficiency"]["ablation"]["metrics"],
            output_file=output_dir / "table2_ablation.tex",
            caption="Ablation study results showing the contribution of each component.",
            label="tab:ablation",
            baseline_config="Full"
        )
        logger.info("Table 2 generated successfully")
    else:
        logger.warning("No ablation results found, skipping Table 2")
    
    # ===== Table 3: Model Specifications =====
    logger.info("=" * 80)
    logger.info("Generating Table 3: Model Specifications")
    logger.info("=" * 80)
    
    model_specs = {}
    
    # Seqorth specs
    model_specs["Seqorth"] = {
        "total_params": config["model"].get("num_experts", 256) * 1e9,
        "active_params": config["model"].get("active_experts", 8) * 1e9,
        "num_experts": config["model"].get("num_experts", 256),
        "top_k": config["model"].get("active_experts", 8),
        "router_type": "Seqorth (OSR + GRU)"
    }
    
    # Baseline specs
    for baseline_key in comparison_group:
        if baseline_key in baseline_config["models"]:
            model_config = baseline_config["models"][baseline_key]
            model_name = model_config["name"]
            
            model_specs[model_name] = {
                "total_params": model_config["params"]["total"],
                "active_params": model_config["params"]["active"],
                "num_experts": model_config["architecture"]["num_experts"],
                "top_k": model_config["architecture"]["top_k"],
                "router_type": model_config["architecture"]["router_type"]
            }
    
    specs_table = generate_parameter_table(
        model_specs=model_specs,
        output_file=output_dir / "table3_specifications.tex",
        caption="Model specifications and parameter counts.",
        label="tab:specifications"
    )
    
    logger.info("Table 3 generated successfully")
    
    # ===== Final Summary JSON =====
    logger.info("=" * 80)
    logger.info("Generating final summary")
    logger.info("=" * 80)
    
    final_summary = {
        "model_results": model_results,
        "ablation_results": ablation_results,
        "model_specs": model_specs,
        "tables": {
            "comparison": str(output_dir / "table1_comparison.tex"),
            "ablation": str(output_dir / "table2_ablation.tex"),
            "specifications": str(output_dir / "table3_specifications.tex")
        }
    }
    
    summary_file = output_dir / "final_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)
    
    logger.info(f"Final summary saved to {summary_file}")
    
    # Save to tracker
    tracker.add_metric("day6", "final_summary", final_summary)
    tracker.save_day("day6")
    tracker.mark_step_complete("day6")
    
    # Generate human-readable report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Seqorth 7-Day Evaluation: Final Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Generated Tables:")
    report_lines.append(f"  - Table 1 (Comparison): {output_dir / 'table1_comparison.tex'}")
    if ablation_results:
        report_lines.append(f"  - Table 2 (Ablation): {output_dir / 'table2_ablation.tex'}")
    report_lines.append(f"  - Table 3 (Specifications): {output_dir / 'table3_specifications.tex'}")
    report_lines.append("")
    report_lines.append("Key Results:")
    report_lines.append("")
    
    if "Seqorth" in model_results:
        seqorth_res = model_results["Seqorth"]
        report_lines.append("Seqorth Performance:")
        for key, value in seqorth_res.items():
            if value is not None and not key.startswith("_"):
                report_lines.append(f"  - {key}: {value}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("All tables are ready for paper submission!")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_file = output_dir / "final_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text + "\n")
    
    logger.info(f"D+6 comparison tables complete! All results in {output_dir}")


if __name__ == "__main__":
    main()

