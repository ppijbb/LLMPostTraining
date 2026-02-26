#!/usr/bin/env python3
"""
D+5: Efficiency & Ablation Study

1. vLLM throughput measurement (inference speed)
2. Ablation study (No GRU, No OSR, No Explicit Bias)
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.routing_benchmarks.utils import (
    load_seqorth_checkpoint,
    load_baseline_model,
    MetricTracker,
    plot_throughput_comparison,
    generate_ablation_table
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def measure_vllm_throughput(
    model_path: str,
    input_lengths: List[int],
    batch_sizes: List[int],
    num_runs: int = 100,
    warmup_runs: int = 10,
    tensor_parallel_size: int = 1
) -> Dict[str, Any]:
    """
    Measure inference throughput using vLLM.
    
    Args:
        model_path: Path to model
        input_lengths: List of input sequence lengths to test
        batch_sizes: List of batch sizes to test
        num_runs: Number of runs for measurement
        warmup_runs: Number of warmup runs
        tensor_parallel_size: Tensor parallelism degree
    
    Returns:
        Dictionary of throughput metrics
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM not installed. Install with: pip install vllm")
        return {}
    
    logger.info(f"Initializing vLLM with model: {model_path}")
    
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        return {}
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        top_p=1.0
    )
    
    results = {
        "throughput_by_config": {},
        "ttft_by_config": {},
        "tpot_by_config": {}
    }
    
    for input_length in input_lengths:
        for batch_size in batch_sizes:
            config_key = f"len{input_length}_bs{batch_size}"
            logger.info(f"Testing config: {config_key}")
            
            # Generate dummy prompts
            dummy_prompt = "Hello world! " * (input_length // 3)
            prompts = [dummy_prompt] * batch_size
            
            # Warmup
            for _ in range(warmup_runs):
                _ = llm.generate(prompts, sampling_params)
            
            # Measure
            total_tokens = 0
            total_time = 0
            ttfts = []
            tpots = []
            
            for _ in range(num_runs):
                start_time = time.time()
                outputs = llm.generate(prompts, sampling_params)
                end_time = time.time()
                
                batch_time = end_time - start_time
                total_time += batch_time
                
                # Count tokens
                for output in outputs:
                    total_tokens += len(output.outputs[0].token_ids)
                    
                    # Estimate TTFT and TPOT (approximation)
                    if len(output.outputs[0].token_ids) > 0:
                        ttft = batch_time / len(outputs)  # Approximate
                        tpot = batch_time / total_tokens if total_tokens > 0 else 0
                        ttfts.append(ttft * 1000)  # Convert to ms
                        tpots.append(tpot * 1000) # Convert to ms
            
            # Calculate metrics
            throughput = total_tokens / total_time
            avg_ttft = np.mean(ttfts) if ttfts else 0
            avg_tpot = np.mean(tpots) if tpots else 0
            
            results["throughput_by_config"][config_key] = throughput
            results["ttft_by_config"][config_key] = avg_ttft
            results["tpot_by_config"][config_key] = avg_tpot
            
            logger.info(f"  Throughput: {throughput:.1f} tokens/sec")
            logger.info(f"  TTFT: {avg_ttft:.1f} ms")
            logger.info(f"  TPOT: {avg_tpot:.1f} ms")
    
    # Calculate averages
    results["avg_throughput"] = np.mean(list(results["throughput_by_config"].values()))
    results["avg_ttft"] = np.mean(list(results["ttft_by_config"].values()))
    results["avg_tpot"] = np.mean(list(results["tpot_by_config"].values()))
    
    return results


def run_ablation_study(
    ablation_configs: Dict[str, str],
    metrics: List[str],
    config: Dict[str, Any],
    tracker: MetricTracker
) -> Dict[str, Dict[str, Any]]:
    """
    Run ablation study by evaluating different configurations.
    
    Args:
        ablation_configs: Dict mapping config names to checkpoint paths
        metrics: List of metrics to evaluate
        config: Evaluation configuration
        tracker: Metric tracker
    
    Returns:
        Dictionary of ablation results
    """
    from day0_sanity_check import evaluate_perplexity
    
    results = {}
    
    for config_name, checkpoint_path in ablation_configs.items():
        if checkpoint_path is None:
            # Use main checkpoint
            checkpoint_path = config["model"]["checkpoint_path"]
        
        logger.info(f"Evaluating ablation config: {config_name}")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            logger.warning(f"  Checkpoint not found, skipping: {checkpoint_path}")
            continue
        
        config_results = {}
        
        try:
            # Load model
            model, tokenizer = load_seqorth_checkpoint(
                checkpoint_path,
                device="cuda",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # Evaluate perplexity
            if "perplexity" in metrics:
                ppl_result = evaluate_perplexity(
                    model=model,
                    tokenizer=tokenizer,
                    dataset_name="wikitext",
                    max_samples=config["day5_efficiency"]["ablation"]["ablation_dataset_size"]
                )
                config_results["perplexity"] = ppl_result["perplexity"]
            
            # Evaluate other metrics (simplified - would use lm-eval for full evaluation)
            for metric in metrics:
                if metric not in config_results and metric != "perplexity":
                    # Placeholder for other metrics
                    config_results[metric] = None
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to evaluate {config_name}: {e}")
            config_results["error"] = str(e)
        
        results[config_name] = config_results
        tracker.add_metric("day5", f"ablation_{config_name}", config_results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="D+5: Efficiency & Ablation")
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
        default="./evaluation_results/day5",
        help="Output directory"
    )
    parser.add_argument(
        "--skip_throughput",
        action="store_true",
        help="Skip throughput measurement"
    )
    parser.add_argument(
        "--skip_ablation",
        action="store_true",
        help="Skip ablation study"
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip baseline throughput measurement"
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
    
    checkpoint_path = args.checkpoint or config["model"]["checkpoint_path"]
    
    # ===== Step 1: vLLM Throughput Measurement =====
    if not args.skip_throughput:
        logger.info("=" * 80)
        logger.info("Measuring Seqorth throughput with vLLM")
        logger.info("=" * 80)
        
        throughput_config = config["day5_efficiency"]["throughput"]
        
        seqorth_metrics = measure_vllm_throughput(
            model_path=checkpoint_path,
            input_lengths=throughput_config["input_lengths"],
            batch_sizes=throughput_config["batch_sizes"],
            num_runs=throughput_config["num_runs"],
            warmup_runs=throughput_config.get("warmup_runs", 10),
            tensor_parallel_size=config["compute"]["tensor_parallel_size"]
        )
        
        tracker.add_metric("day5", "throughput_seqorth", seqorth_metrics)
        
        # Measure baseline throughput
        baseline_metrics = {}
        
        if not args.skip_baselines:
            baseline_config_file = Path(args.config).parent / "baseline_models.yaml"
            with open(baseline_config_file, 'r') as f:
                baseline_config = yaml.safe_load(f)
            
            comparison_group = baseline_config["comparison_groups"]["similar_active_params"]
            
            for baseline_key in comparison_group:
                model_config = baseline_config["models"][baseline_key]
                model_name = model_config["name"]
                model_path = model_config["hf_path"]
                
                logger.info(f"Measuring {model_name} throughput...")
                
                try:
                    metrics = measure_vllm_throughput(
                        model_path=model_path,
                        input_lengths=throughput_config["input_lengths"],
                        batch_sizes=throughput_config["batch_sizes"],
                        num_runs=throughput_config["num_runs"] // 2,  # Faster for baselines
                        tensor_parallel_size=config["compute"]["tensor_parallel_size"]
                    )
                    baseline_metrics[model_name] = metrics
                    tracker.add_metric("day5", f"throughput_{model_name}", metrics)
                    
                except Exception as e:
                    logger.error(f"Failed to measure {model_name}: {e}")
        
        # Generate comparison plot
        if seqorth_metrics and baseline_metrics:
            logger.info("Generating throughput comparison plot...")
            
            seqorth_plot_metrics = {
                "tokens_per_sec": seqorth_metrics.get("avg_throughput", 0),
                "ttft_ms": seqorth_metrics.get("avg_ttft", 0)
            }
            
            baseline_plot_metrics = {
                name: {
                    "tokens_per_sec": metrics.get("avg_throughput", 0),
                    "ttft_ms": metrics.get("avg_ttft", 0)
                }
                for name, metrics in baseline_metrics.items()
            }
            
            plot_throughput_comparison(
                seqorth_metrics=seqorth_plot_metrics,
                baseline_metrics=baseline_plot_metrics,
                output_file=output_dir / "throughput_comparison.png"
            )
    
    # ===== Step 2: Ablation Study =====
    if not args.skip_ablation:
        logger.info("=" * 80)
        logger.info("Running ablation study")
        logger.info("=" * 80)
        
        ablation_config = config["day5_efficiency"]["ablation"]
        
        # Build ablation configs
        ablation_configs = {}
        for config_spec in ablation_config["configs"]:
            config_name = config_spec["name"]
            config_checkpoint = config_spec.get("checkpoint")
            ablation_configs[config_name] = config_checkpoint
        
        # Run ablation
        ablation_results = run_ablation_study(
            ablation_configs=ablation_configs,
            metrics=ablation_config["metrics"],
            config=config,
            tracker=tracker
        )
        
        # Generate ablation table
        logger.info("Generating ablation table...")
        
        ablation_table = generate_ablation_table(
            ablation_results=ablation_results,
            metrics=ablation_config["metrics"],
            output_file=output_dir / "ablation_table.tex",
            baseline_config="Full"
        )
        
        logger.info("Ablation table generated")
    
    # Save results
    tracker.save_day("day5")
    tracker.mark_step_complete("day5")
    
    # Export summary
    summary = tracker.export_summary(output_dir / "efficiency_report.json")
    
    logger.info(f"D+5 efficiency & ablation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

