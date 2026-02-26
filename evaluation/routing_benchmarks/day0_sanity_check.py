#!/usr/bin/env python3
"""
D-Day: Sanity Check - Training Dynamics and Perplexity Evaluation

This script performs the initial "GO/NO-GO" evaluation:
1. Extract and visualize training curves (CV, MaxVio, Loss)
2. Measure validation perplexity on multiple datasets
3. Generate sanity check report with GO/NO-GO decision
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.routing_benchmarks.utils import (
    load_seqorth_checkpoint,
    prepare_model_for_eval,
    extract_cv_curve,
    extract_maxvio_curve,
    extract_loss_curve,
    extract_orthogonality_curve,
    find_explicit_bias_start_step,
    MetricTracker,
    plot_cv_curve,
    plot_maxvio_curve,
    plot_loss_curve,
    create_multi_curve_figure
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_name: str,
    dataset_split: str = "validation",
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    max_length: int = 2048,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset_name: Dataset name (e.g., "wikitext", "pile", "c4")
        dataset_split: Dataset split
        max_samples: Maximum number of samples (None for all)
        batch_size: Batch size
        max_length: Maximum sequence length
        device: Device to use
    
    Returns:
        Dictionary with perplexity and loss
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    logger.info(f"Evaluating perplexity on {dataset_name} ({dataset_split})")
    
    # Load dataset
    try:
        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-v1", split=dataset_split)
        elif dataset_name == "pile":
            dataset = load_dataset("monology/pile-uncopyrighted", split=dataset_split, streaming=True)
            if max_samples:
                dataset = dataset.take(max_samples)
        elif dataset_name == "c4":
            dataset = load_dataset("c4", "en", split=dataset_split, streaming=True)
            if max_samples:
                dataset = dataset.take(max_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return {"perplexity": float('inf'), "loss": float('inf'), "error": str(e)}
    
    # Limit samples
    if max_samples and not hasattr(dataset, '__iter__'):
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Tokenize and evaluate
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        texts = []
        for item in dataset:
            text = item.get("text", item.get("content", ""))
            if text.strip():
                texts.append(text)
            
            if len(texts) >= batch_size:
                # Tokenize batch
                try:
                    inputs = tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(device)
                    
                    # Forward pass
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    # Accumulate
                    num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing batch: {e}")
                
                texts = []
                
                if max_samples and num_batches * batch_size >= max_samples:
                    break
        
        # Process remaining texts
        if texts:
            try:
                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Error processing final batch: {e}")
    
    if total_tokens == 0:
        logger.error("No tokens processed")
        return {"perplexity": float('inf'), "loss": float('inf'), "error": "No tokens processed"}
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"  Perplexity: {perplexity:.2f}, Loss: {avg_loss:.4f}, Tokens: {total_tokens:,}")
    
    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "num_tokens": total_tokens,
        "num_batches": num_batches
    }


def generate_sanity_check_report(
    metrics: Dict[str, Any],
    thresholds: Dict[str, float],
    output_file: Path
) -> Dict[str, Any]:
    """
    Generate GO/NO-GO report based on sanity check metrics.
    
    Args:
        metrics: Dictionary of collected metrics
        thresholds: Dictionary of threshold values
        output_file: Path to save report
    
    Returns:
        Report dictionary with GO/NO-GO decision
    """
    report = {
        "timestamp": torch.utils.data.get_worker_info(),
        "decision": "GO",
        "checks": [],
        "metrics": metrics,
        "thresholds": thresholds
    }
    
    # Check 1: CV should be low
    cv_final = metrics.get("cv_final")
    if cv_final is not None:
        cv_threshold = thresholds.get("max_cv", 0.3)
        cv_pass = cv_final < cv_threshold
        report["checks"].append({
            "name": "CV (Coefficient of Variation)",
            "value": cv_final,
            "threshold": f"< {cv_threshold}",
            "passed": cv_pass,
            "importance": "CRITICAL"
        })
        if not cv_pass:
            report["decision"] = "NO-GO"
    
    # Check 2: MaxVio should be low
    maxvio_final = metrics.get("maxvio_final")
    if maxvio_final is not None:
        maxvio_threshold = thresholds.get("max_maxvio", 0.15)
        maxvio_pass = maxvio_final < maxvio_threshold
        report["checks"].append({
            "name": "MaxVio (Constraint Violation)",
            "value": maxvio_final,
            "threshold": f"< {maxvio_threshold}",
            "passed": maxvio_pass,
            "importance": "CRITICAL"
        })
        if not maxvio_pass:
            report["decision"] = "NO-GO"
    
    # Check 3: Perplexity should be reasonable
    ppl_wikitext = metrics.get("perplexity_wikitext")
    if ppl_wikitext is not None:
        ppl_pass = ppl_wikitext < 1000  # Very high threshold - just sanity check
        report["checks"].append({
            "name": "Perplexity (WikiText-103)",
            "value": ppl_wikitext,
            "threshold": "< 1000 (sanity)",
            "passed": ppl_pass,
            "importance": "HIGH"
        })
        if not ppl_pass:
            report["decision"] = "NO-GO"
    
    # Generate text report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Seqorth D-Day Sanity Check Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"DECISION: {report['decision']}")
    report_lines.append("")
    report_lines.append("Checks:")
    report_lines.append("-" * 80)
    
    for check in report["checks"]:
        status = "✓ PASS" if check["passed"] else "✗ FAIL"
        report_lines.append(f"  [{status}] {check['name']}")
        report_lines.append(f"    Value: {check['value']:.4f if isinstance(check['value'], float) else check['value']}")
        report_lines.append(f"    Threshold: {check['threshold']}")
        report_lines.append(f"    Importance: {check['importance']}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    if report["decision"] == "GO":
        report_lines.append("✓ Model is ready for paper evaluation!")
    else:
        report_lines.append("✗ Model has issues - review training before proceeding")
    
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    # Also save JSON
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Sanity check report saved to {output_file}")
    print("\n" + report_text + "\n")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="D-Day: Seqorth Sanity Check")
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
        default="./evaluation_results/day0",
        help="Output directory"
    )
    parser.add_argument(
        "--skip_wandb",
        action="store_true",
        help="Skip WandB extraction (use checkpoint only)"
    )
    parser.add_argument(
        "--skip_ppl",
        action="store_true",
        help="Skip perplexity evaluation"
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
        use_timestamp=False  # Use fixed directory for day0
    )
    
    # Extract checkpoint path
    checkpoint_path = args.checkpoint or config["model"]["checkpoint_path"]
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # ===== Step 1: Extract Training Curves from WandB =====
    if not args.skip_wandb and config["wandb"]["enabled"]:
        logger.info("Extracting training curves from WandB...")
        
        run_id = config["wandb"]["run_id"]
        project = config["wandb"]["project"]
        entity = config["wandb"].get("entity")
        
        try:
            # Extract CV curve
            cv_df = extract_cv_curve(run_id, project, entity)
            tracker.add_metric("day0", "cv_curve", cv_df.to_dict('list'))
            
            # Extract MaxVio curve
            maxvio_df = extract_maxvio_curve(run_id, project, entity)
            tracker.add_metric("day0", "maxvio_curve", maxvio_df.to_dict('list'))
            
            # Extract loss curve
            loss_df = extract_loss_curve(run_id, project, entity)
            tracker.add_metric("day0", "loss_curve", loss_df.to_dict('list'))
            
            # Extract orthogonality curve
            ortho_df = extract_orthogonality_curve(run_id, project, entity)
            tracker.add_metric("day0", "orthogonality_curve", ortho_df.to_dict('list'))
            
            # Find explicit bias start step
            bias_step = find_explicit_bias_start_step(run_id, project, entity)
            tracker.add_metric("day0", "explicit_bias_step", bias_step)
            
            logger.info("Training curves extracted successfully")
            
            # Generate visualizations
            logger.info("Generating training curve visualizations...")
            
            # CV curve
            plot_cv_curve(
                cv_df,
                output_file=output_dir / "cv_curve.png",
                explicit_bias_step=bias_step
            )
            
            # MaxVio curve
            plot_maxvio_curve(
                maxvio_df,
                output_file=output_dir / "maxvio_curve.png"
            )
            
            # Loss curve
            plot_loss_curve(
                loss_df,
                output_file=output_dir / "loss_curve.png"
            )
            
            # Multi-curve figure
            curves = {
                "CV": cv_df,
                "MaxVio": maxvio_df,
                "Loss": loss_df
            }
            if not ortho_df.empty:
                curves["Orthogonality"] = ortho_df
            
            create_multi_curve_figure(
                curves,
                output_file=output_dir / "training_dynamics.png",
                title="Seqorth Training Dynamics"
            )
            
            # Store final values for GO/NO-GO
            cv_final = cv_df["cv"].iloc[-1] if not cv_df.empty else None
            maxvio_final = maxvio_df["maxvio"].iloc[-1] if not maxvio_df.empty else None
            
            tracker.add_metric("day0", "cv_final", cv_final)
            tracker.add_metric("day0", "maxvio_final", maxvio_final)
            
        except Exception as e:
            logger.error(f"Failed to extract WandB curves: {e}")
            if not args.skip_wandb:
                logger.warning("Continuing without WandB data...")
    
    # ===== Step 2: Evaluate Perplexity =====
    if not args.skip_ppl:
        logger.info("Evaluating perplexity...")
        
        # Load model
        logger.info("Loading model...")
        model, tokenizer = load_seqorth_checkpoint(
            checkpoint_path,
            device="cuda",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = prepare_model_for_eval(model)
        
        # Evaluate on each dataset
        ppl_config = config["day0_sanity_check"]["ppl_datasets"]
        
        for dataset_config in ppl_config:
            dataset_name = dataset_config["name"]
            dataset_split = dataset_config.get("split", "validation")
            max_samples = dataset_config.get("max_samples")
            
            ppl_result = evaluate_perplexity(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                max_samples=max_samples,
                batch_size=config["compute"]["batch_size_per_gpu"],
                max_length=config["compute"]["max_seq_length"]
            )
            
            tracker.add_metric("day0", f"perplexity_{dataset_name}", ppl_result["perplexity"])
            tracker.add_metric("day0", f"loss_{dataset_name}", ppl_result["loss"])
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # ===== Step 3: Generate Sanity Check Report =====
    logger.info("Generating sanity check report...")
    
    metrics = {
        "cv_final": tracker.get_metric("day0", "cv_final"),
        "maxvio_final": tracker.get_metric("day0", "maxvio_final"),
        "perplexity_wikitext": tracker.get_metric("day0", "perplexity_wikitext-103"),
        "perplexity_pile": tracker.get_metric("day0", "perplexity_pile"),
        "perplexity_c4": tracker.get_metric("day0", "perplexity_c4"),
    }
    
    thresholds = config["day0_sanity_check"]["go_nogo_thresholds"]
    
    report = generate_sanity_check_report(
        metrics=metrics,
        thresholds=thresholds,
        output_file=output_dir / "sanity_check_report.txt"
    )
    
    tracker.add_metric("day0", "sanity_check_report", report)
    
    # Save all results
    tracker.save_day("day0")
    tracker.mark_step_complete("day0")
    
    logger.info(f"D-Day sanity check complete! Results saved to {output_dir}")
    logger.info(f"Decision: {report['decision']}")
    
    # Exit with appropriate code
    sys.exit(0 if report["decision"] == "GO" else 1)


if __name__ == "__main__":
    main()

