#!/usr/bin/env python3
"""
D+1~2: Standard Benchmarks using lm-evaluation-harness

Evaluate Seqorth and baseline models on standard benchmarks:
- Knowledge: MMLU
- Reasoning: GSM8K, ARC-Challenge
- Commonsense: HellaSwag, Winogrande
- Coding: HumanEval, MBPP
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import json
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.routing_benchmarks.utils import (
    load_seqorth_checkpoint,
    load_baseline_model,
    prepare_model_for_eval,
    MetricTracker
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_lm_eval(
    model_path: str,
    tasks: List[str],
    output_dir: Path,
    num_fewshot: int = 5,
    batch_size: str = "auto",
    device: str = "cuda",
    limit: Optional[int] = None,
    trust_remote_code: bool = True
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness on specified tasks.
    
    Args:
        model_path: Path to model or HF model name
        tasks: List of task names
        output_dir: Directory to save results
        num_fewshot: Number of few-shot examples
        batch_size: Batch size ("auto" or integer)
        device: Device to use
        limit: Limit number of examples (None for all)
        trust_remote_code: Whether to trust remote code
    
    Returns:
        Dictionary of results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code={trust_remote_code}",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--device", device,
        "--output_path", str(output_dir)
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    logger.info(f"Running lm-eval with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("lm-eval completed successfully")
        logger.debug(result.stdout)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"lm-eval failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        raise
    
    # Parse results
    results_file = output_dir / "results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    else:
        logger.warning(f"Results file not found: {results_file}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="D+1~2: Standard Benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation_config.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to Seqorth checkpoint (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results/day1_2",
        help="Output directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to evaluate (default: Seqorth + baselines)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Tasks to run (default: all from config)"
    )
    parser.add_argument(
        "--skip_seqorth",
        action="store_true",
        help="Skip Seqorth evaluation"
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip baseline evaluation"
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
    
    # Determine tasks to run
    if args.tasks:
        task_categories = {"custom": args.tasks}
    else:
        task_categories = config["day1_2_benchmarks"]["tasks"]
    
    # Shot configuration
    shot_config = config["day1_2_benchmarks"]["shot_config"]
    batch_size = config["day1_2_benchmarks"]["batch_size"]
    limit = config["day1_2_benchmarks"].get("limit")
    
    # ===== Evaluate Seqorth =====
    if not args.skip_seqorth:
        logger.info("=" * 80)
        logger.info("Evaluating Seqorth")
        logger.info("=" * 80)
        
        checkpoint_path = args.checkpoint or config["model"]["checkpoint_path"]
        model_name = "Seqorth"
        
        model_output_dir = output_dir / "seqorth"
        
        all_results = {}
        
        for category, tasks in task_categories.items():
            logger.info(f"Running {category} tasks: {tasks}")
            
            num_fewshot = shot_config.get(category, 5)
            
            category_output_dir = model_output_dir / category
            
            try:
                results = run_lm_eval(
                    model_path=checkpoint_path,
                    tasks=tasks,
                    output_dir=category_output_dir,
                    num_fewshot=num_fewshot,
                    batch_size=batch_size,
                    limit=limit
                )
                
                # Extract key metrics
                for task in tasks:
                    if task in results.get("results", {}):
                        task_results = results["results"][task]
                        # Get primary metric (usually acc, exact_match, or pass@1)
                        metric_key = None
                        for key in ["acc", "acc_norm", "exact_match", "pass@1"]:
                            if key in task_results:
                                metric_key = key
                                break
                        
                        if metric_key:
                            all_results[task] = task_results[metric_key]
                        else:
                            all_results[task] = task_results
                
            except Exception as e:
                logger.error(f"Failed to evaluate {category}: {e}")
        
        # Save Seqorth results
        tracker.add_model_result("day1_2", model_name, all_results)
        
        logger.info(f"Seqorth results: {all_results}")
    
    # ===== Evaluate Baselines =====
    if not args.skip_baselines and config["day1_2_benchmarks"].get("run_baselines", True):
        logger.info("=" * 80)
        logger.info("Evaluating Baseline Models")
        logger.info("=" * 80)
        
        # Load baseline configs
        baseline_config_file = Path(args.config).parent / "baseline_models.yaml"
        with open(baseline_config_file, 'r') as f:
            baseline_config = yaml.safe_load(f)
        
        # Determine which baselines to run
        if args.models:
            baseline_names = args.models
        else:
            comparison_group = config["day1_2_benchmarks"].get("comparison_group", "similar_active_params")
            baseline_names = baseline_config["comparison_groups"].get(comparison_group, [])
        
        for baseline_key in baseline_names:
            if baseline_key not in baseline_config["models"]:
                logger.warning(f"Baseline {baseline_key} not found in config, skipping")
                continue
            
            model_config = baseline_config["models"][baseline_key]
            model_name = model_config["name"]
            model_path = model_config["hf_path"]
            
            logger.info(f"Evaluating {model_name}...")
            
            model_output_dir = output_dir / baseline_key
            
            all_results = {}
            
            for category, tasks in task_categories.items():
                logger.info(f"Running {category} tasks: {tasks}")
                
                num_fewshot = shot_config.get(category, 5)
                category_output_dir = model_output_dir / category
                
                try:
                    results = run_lm_eval(
                        model_path=model_path,
                        tasks=tasks,
                        output_dir=category_output_dir,
                        num_fewshot=num_fewshot,
                        batch_size=batch_size,
                        limit=limit
                    )
                    
                    # Extract key metrics
                    for task in tasks:
                        if task in results.get("results", {}):
                            task_results = results["results"][task]
                            metric_key = None
                            for key in ["acc", "acc_norm", "exact_match", "pass@1"]:
                                if key in task_results:
                                    metric_key = key
                                    break
                            
                            if metric_key:
                                all_results[task] = task_results[metric_key]
                            else:
                                all_results[task] = task_results
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name} on {category}: {e}")
            
            # Save baseline results
            tracker.add_model_result("day1_2", model_name, all_results)
            
            logger.info(f"{model_name} results: {all_results}")
    
    # ===== Generate Summary =====
    logger.info("Generating comparison summary...")
    
    # Get all model results
    summary = {}
    for model_name in ["Seqorth"] + [baseline_config["models"][k]["name"] 
                                      for k in baseline_names 
                                      if k in baseline_config["models"]]:
        results = tracker.get_model_result("day1_2", model_name)
        if results:
            summary[model_name] = results
    
    # Save summary
    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Comparison summary saved to {summary_file}")
    
    # Save tracker
    tracker.save_day("day1_2")
    tracker.mark_step_complete("day1_2")
    
    logger.info(f"D+1~2 benchmarks complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

