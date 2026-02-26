#!/usr/bin/env python3
"""
Complete Benchmark Suite for Seqorth MoE Paper Evaluation

Runs all standard benchmarks:
- Perplexity (held-out test set)
- MMLU, GSM8K, ARC-Easy/Challenge, HellaSwag, TruthfulQA, WinoGrande, PIQA, HumanEval

Uses lm-evaluation-harness for comprehensive evaluation.
"""

import os
import sys
import json
import torch
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable torch.compile to avoid issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch.compiler.disable()
torch._dynamo.reset()


def check_lm_eval_harness():
    """Check if lm-evaluation-harness is installed"""
    try:
        import lm_eval
        return True
    except ImportError:
        return False


def run_lm_eval_harness(
    model_path: str,
    tasks: List[str],
    output_dir: str,
    batch_size: int = 8,
    device: str = "cuda",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness benchmarks
    
    Args:
        model_path: Path to model or model identifier
        tasks: List of task names (e.g., ['mmlu', 'gsm8k', 'arc_easy'])
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        device: Device to use
        limit: Limit number of examples per task (None = all)
        
    Returns:
        Dictionary with results
    """
    if not check_lm_eval_harness():
        raise ImportError(
            "lm-evaluation-harness not installed. Install with: pip install lm-eval>=0.4.0"
        )
    
    import lm_eval
    from lm_eval import tasks
    from lm_eval.models import huggingface
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--device", device,
        "--output_path", str(output_dir / "lm_eval_results.json"),
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    print(f"Running lm-evaluation-harness with command:")
    print(" ".join(cmd))
    print()
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running lm-evaluation-harness:")
        print(result.stderr)
        raise RuntimeError(f"lm-evaluation-harness failed with code {result.returncode}")
    
    # Load results
    results_path = output_dir / "lm_eval_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    else:
        print(f"Warning: Results file not found at {results_path}")
        return {}


def run_perplexity_evaluation(
    model,
    tokenizer,
    eval_dataset: List[str],
    device: str = "cuda",
    max_samples: int = 1000,
) -> Dict[str, float]:
    """Evaluate perplexity on held-out dataset"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for i, text in enumerate(tqdm(eval_dataset[:max_samples], desc="Evaluating perplexity")):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                num_tokens = inputs['input_ids'].numel()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    if total_tokens == 0:
        return {'perplexity': float('inf'), 'loss': float('inf')}
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return {'perplexity': perplexity, 'loss': total_loss / total_tokens}


def run_human_eval(
    model,
    tokenizer,
    device: str = "cuda",
) -> Dict[str, float]:
    """Run HumanEval code generation benchmark"""
    try:
        from human_eval import HumanEval
        from human_eval.data import HUMAN_EVAL
        
        human_eval = HumanEval()
        results = human_eval.evaluate(
            model,
            tokenizer,
            device=device,
        )
        return results
    except ImportError:
        print("Warning: human-eval not installed. Install with: pip install human-eval")
        return {}
    except Exception as e:
        print(f"Error running HumanEval: {e}")
        return {}


def format_results_table(results: Dict[str, Any]) -> str:
    """Format results as markdown table"""
    lines = ["# Benchmark Results", "", "| Task | Metric | Score |", "|------|--------|-------|"]
    
    for task_name, task_results in results.items():
        if isinstance(task_results, dict):
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    lines.append(f"| {task_name} | {metric_name} | {metric_value:.4f} |")
                else:
                    lines.append(f"| {task_name} | {metric_name} | {metric_value} |")
        else:
            lines.append(f"| {task_name} | score | {task_results} |")
    
    return "\n".join(lines)


def generate_latex_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX table for paper"""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Task & Metric & Score \\\\",
        "\\midrule",
    ]
    
    for task_name, task_results in results.items():
        if isinstance(task_results, dict):
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    lines.append(f"{task_name} & {metric_name} & {metric_value:.4f} \\\\")
                else:
                    lines.append(f"{task_name} & {metric_name} & {metric_value} \\\\")
        else:
            lines.append(f"{task_name} & score & {results} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Benchmark Results}",
        "\\label{tab:benchmark_results}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Complete Benchmark Suite for Seqorth MoE")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model identifier",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "mmlu",
            "gsm8k",
            "arc_easy",
            "arc_challenge",
            "hellaswag",
            "truthfulqa",
            "winogrande",
            "piqa",
            "human_eval",
        ],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--perplexity_dataset",
        type=str,
        default=None,
        help="Path to text file for perplexity evaluation (one text per line)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task",
    )
    parser.add_argument(
        "--skip_lm_eval",
        action="store_true",
        help="Skip lm-evaluation-harness (use only custom evaluations)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run lm-evaluation-harness benchmarks
    if not args.skip_lm_eval:
        print("=" * 60)
        print("Running lm-evaluation-harness benchmarks")
        print("=" * 60)
        
        # Filter out tasks that need special handling
        lm_eval_tasks = [t for t in args.tasks if t != "human_eval"]
        
        if lm_eval_tasks:
            try:
                lm_eval_results = run_lm_eval_harness(
                    model_path=args.model_path,
                    tasks=lm_eval_tasks,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    device=args.device,
                    limit=args.limit,
                )
                results.update(lm_eval_results)
            except Exception as e:
                print(f"Error running lm-evaluation-harness: {e}")
                import traceback
                traceback.print_exc()
    
    # Run HumanEval separately (if requested)
    if "human_eval" in args.tasks:
        print("\n" + "=" * 60)
        print("Running HumanEval")
        print("=" * 60)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            
            human_eval_results = run_human_eval(model, tokenizer, device=args.device)
            if human_eval_results:
                results["human_eval"] = human_eval_results
        except Exception as e:
            print(f"Error running HumanEval: {e}")
            import traceback
            traceback.print_exc()
    
    # Run perplexity evaluation (if dataset provided)
    if args.perplexity_dataset:
        print("\n" + "=" * 60)
        print("Running Perplexity Evaluation")
        print("=" * 60)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load dataset
            with open(args.perplexity_dataset, 'r') as f:
                eval_dataset = [line.strip() for line in f if line.strip()]
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            
            ppl_results = run_perplexity_evaluation(
                model, tokenizer, eval_dataset, device=args.device
            )
            results["perplexity"] = ppl_results
        except Exception as e:
            print(f"Error running perplexity evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json_path = output_dir / f"benchmark_results_{timestamp}.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown table
    markdown_table = format_results_table(results)
    markdown_path = output_dir / f"benchmark_results_{timestamp}.md"
    with open(markdown_path, 'w') as f:
        f.write(markdown_table)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    latex_path = output_dir / f"benchmark_results_{timestamp}.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    
    print("\n" + "=" * 60)
    print("Benchmark Evaluation Complete")
    print("=" * 60)
    print(f"Results saved to:")
    print(f"  - JSON: {results_json_path}")
    print(f"  - Markdown: {markdown_path}")
    print(f"  - LaTeX: {latex_path}")
    print("\nResults Summary:")
    print(markdown_table)


if __name__ == "__main__":
    main()

