#!/usr/bin/env python3
"""
Efficiency Measurement Tool for Seqorth MoE

Measures:
- tokens/s (throughput)
- latency (per token, mean/p50/p95/p99)
- memory usage (peak/avg)
- FLOPs (active/total)
- wall-clock time to target PPL (training efficiency curve)
"""

import os
import sys
import torch
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import psutil
import GPUtil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable torch.compile
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True


def get_memory_usage():
    """Get current memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e9


def measure_forward_throughput(
    model,
    tokenizer,
    input_text: str,
    batch_sizes: List[int] = [1, 4, 8, 16],
    seq_length: int = 512,
    num_runs: int = 50,
    warmup_runs: int = 10,
    device: str = "cuda",
) -> Dict[int, Dict[str, float]]:
    """
    Measure forward pass throughput for different batch sizes
    
    Returns:
        Dictionary mapping batch_size -> {tokens_per_sec, latency_ms, memory_gb}
    """
    model.eval()
    results = {}
    
    # Prepare input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_length,
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    for batch_size in batch_sizes:
        # Create batched input
        batched_input_ids = input_ids.repeat(batch_size, 1)
        if attention_mask is not None:
            batched_attention_mask = attention_mask.repeat(batch_size, 1)
        else:
            batched_attention_mask = None
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_mask,
                )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Measure
        latencies = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = model(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_mask,
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # ms
                latencies.append(latency)
                
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9
                    memory_usage.append(peak_memory)
                else:
                    memory_usage.append(get_memory_usage())
        
        # Calculate statistics
        latencies = np.array(latencies)
        tokens_per_batch = batched_input_ids.numel()
        
        avg_latency_ms = np.mean(latencies)
        tokens_per_sec = (tokens_per_batch / avg_latency_ms) * 1000  # Convert ms to sec
        
        results[batch_size] = {
            "tokens_per_sec": tokens_per_sec,
            "latency_ms_mean": avg_latency_ms,
            "latency_ms_p50": np.percentile(latencies, 50),
            "latency_ms_p95": np.percentile(latencies, 95),
            "latency_ms_p99": np.percentile(latencies, 99),
            "memory_gb_peak": np.max(memory_usage) if memory_usage else 0,
            "memory_gb_avg": np.mean(memory_usage) if memory_usage else 0,
        }
        
        print(f"Batch size {batch_size}: {tokens_per_sec:.2f} tokens/s, {avg_latency_ms:.2f} ms latency")
    
    return results


def measure_generation_latency(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 32,
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure generation latency with percentiles
    
    Returns:
        Dictionary with latency statistics
    """
    model.eval()
    
    # Prepare input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    latencies = []
    tokens_generated = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Measuring generation latency"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # ms
            num_tokens = outputs.shape[1] - input_ids.shape[1]
            
            latencies.append(latency)
            tokens_generated.append(num_tokens)
    
    latencies = np.array(latencies)
    tokens_generated = np.array(tokens_generated)
    
    # Per-token latency
    per_token_latencies = latencies / tokens_generated
    
    return {
        "total_latency_ms_mean": np.mean(latencies),
        "total_latency_ms_p50": np.percentile(latencies, 50),
        "total_latency_ms_p95": np.percentile(latencies, 95),
        "total_latency_ms_p99": np.percentile(latencies, 99),
        "per_token_latency_ms_mean": np.mean(per_token_latencies),
        "per_token_latency_ms_p50": np.percentile(per_token_latencies, 50),
        "per_token_latency_ms_p95": np.percentile(per_token_latencies, 95),
        "per_token_latency_ms_p99": np.percentile(per_token_latencies, 99),
        "avg_tokens_generated": np.mean(tokens_generated),
        "tokens_per_sec": np.mean(tokens_generated) / (np.mean(latencies) / 1000),
    }


def estimate_flops(
    model,
    input_shape: Tuple[int, int] = (1, 512),
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Estimate FLOPs for the model
    
    Returns:
        Dictionary with FLOP estimates
    """
    try:
        from fvcore.nn import FlopCountMode, flop_count
        
        model.eval()
        dummy_input = torch.randint(
            0, 1000, input_shape, device=device, dtype=torch.long
        )
        
        flop_counter = FlopCountMode(model, dummy_input)
        flops_dict, _ = flop_count(model, (dummy_input,))
        
        total_flops = sum(flops_dict.values())
        
        # Estimate active FLOPs (only for MoE models)
        # This is a simplified estimate - actual calculation depends on routing
        active_flops = total_flops  # Default: assume all FLOPs are active
        
        return {
            "total_flops": total_flops,
            "active_flops": active_flops,
            "flops_ratio": active_flops / total_flops if total_flops > 0 else 0,
        }
    except ImportError:
        print("Warning: fvcore not installed. Install with: pip install fvcore")
        return {}
    except Exception as e:
        print(f"Error estimating FLOPs: {e}")
        return {}


def measure_training_efficiency_curve(
    model_path: str,
    checkpoint_dir: str,
    eval_dataset: List[str],
    target_ppl: float = 10.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Measure wall-clock time to reach target PPL
    
    This requires access to training logs or checkpoints.
    For now, returns a placeholder structure.
    """
    # This would require parsing training logs or loading checkpoints
    # Implementation depends on how training logs are structured
    
    return {
        "target_ppl": target_ppl,
        "wall_clock_time_hours": None,  # Would be calculated from logs
        "steps_to_target": None,
        "tokens_to_target": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure Model Efficiency")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model identifier",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./efficiency_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="The capital of France is",
        help="Input text for measurement",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=512,
        help="Sequence length for forward pass",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=50,
        help="Number of runs for averaging",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--measure_flops",
        action="store_true",
        help="Measure FLOPs (requires fvcore)",
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
    
    results = {}
    
    # Measure forward throughput
    print("\n" + "=" * 60)
    print("Measuring Forward Pass Throughput")
    print("=" * 60)
    forward_results = measure_forward_throughput(
        model,
        tokenizer,
        args.input_text,
        batch_sizes=args.batch_sizes,
        seq_length=args.seq_length,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        device=args.device,
    )
    results["forward_throughput"] = forward_results
    
    # Measure generation latency
    print("\n" + "=" * 60)
    print("Measuring Generation Latency")
    print("=" * 60)
    generation_results = measure_generation_latency(
        model,
        tokenizer,
        args.input_text,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        device=args.device,
    )
    results["generation_latency"] = generation_results
    
    # Estimate FLOPs
    if args.measure_flops:
        print("\n" + "=" * 60)
        print("Estimating FLOPs")
        print("=" * 60)
        flops_results = estimate_flops(model, input_shape=(1, args.seq_length), device=args.device)
        if flops_results:
            results["flops"] = flops_results
    
    # Save results
    results_json_path = output_dir / "efficiency_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Efficiency Measurement Complete")
    print("=" * 60)
    print(f"Results saved to: {results_json_path}")
    print("\nSummary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

