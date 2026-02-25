#!/usr/bin/env python3
"""
Qwen3-VL-30B-A3 Evaluation Script for Routing Benchmarks.
TURBO MODE: Optimized with Batching and DeepSpeed Stage 3 Tiling.
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import logging
import yaml
import numpy as np
import collections
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
import deepspeed
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add workspace to path
# Expect PYTHONPATH to include repo root

from eval.routing_benchmarks.utils.metric_tracker import MetricTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoutingHook:
    def __init__(self):
        self.layer_data = collections.defaultdict(list)
        
    def __call__(self, module, input, output):
        # Qwen3-VL MoE blocks usually return a tuple where one element contains logits
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 3: # [batch, seq, experts]
                    self.capture(module, item)
                    break
        elif isinstance(output, dict):
            if 'router_logits' in output:
                self.capture(module, output['router_logits'])

    def capture(self, module, logits):
        self.layer_data[id(module)].append(logits.detach().cpu())

def calculate_cv(usage_counts):
    if len(usage_counts) == 0: return 0.0
    mu = np.mean(usage_counts)
    if mu == 0: return 0.0
    sigma = np.std(usage_counts)
    return sigma / mu

def calculate_maxvio(usage_counts, target_usage):
    if len(usage_counts) == 0: return 0.0
    violations = (usage_counts - target_usage) / target_usage
    return np.max(np.maximum(0, violations))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./evaluation_results/qwen3_eval")
    args = parser.parse_args()

    # Distributed Init
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = MetricTracker(base_dir=str(output_dir), use_timestamp=True)

    # 1. Load Model with ZeRO-3 Turbo
    model_id = config["model"]["checkpoint_path"]
    logger.info(f"Loading model {model_id} with ZeRO-3 Turbo Config...")

    ds_config = {
        "zero_optimization": config["compute"]["zero_optimization"],
        "aio": config["compute"]["aio"],
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": config["compute"]["batch_size_per_gpu"],
    }

    with deepspeed.zero.Init():
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    # 2. Register Hooks
    hook_helper = RoutingHook()
    target_blocks = config["routing"]["target_blocks"]
    for name, module in model.named_modules():
        if any(target in str(type(module)) for target in target_blocks):
            module.register_forward_hook(hook_helper)
            logger.info(f"Hooked into {name}")

    # 3. TURBO Data Processing (Batching)
    from datasets import load_dataset
    batch_size = config["compute"]["batch_size_per_gpu"]
    max_seq_len = config["compute"]["max_seq_length"]

    # Text Evaluation
    for ds_conf in config["datasets"].get("text", []):
        logger.info(f"Processing Text: {ds_conf['name']}")
        dataset = load_dataset(ds_conf["name"], ds_conf.get("config"), split=ds_conf["split"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for i, batch in enumerate(tqdm(loader, desc=ds_conf["name"], disable=local_rank != 0)):
            texts = [t for t in (batch.get("text") or batch.get("content") or batch.get("question")) if t]
            if not texts: continue
            
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(model.device)
            with torch.no_grad():
                model(**inputs)
            
            if i >= ds_conf.get("max_batches", 50): break

    # VL Evaluation
    for ds_conf in config["datasets"].get("vision_language", []):
        logger.info(f"Processing VL: {ds_conf['name']}")
        dataset = load_dataset(ds_conf["name"], ds_conf.get("config"), split=ds_conf["split"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for i, batch in enumerate(tqdm(loader, desc=ds_conf["name"], disable=local_rank != 0)):
            # Handle mixed batch items (Assume standard HF dataset return)
            prompts = batch.get("question") or batch.get("text")
            images = batch.get("image")
            
            if prompts is None or images is None: continue
            
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                model(**inputs)
            
            if i >= ds_conf.get("max_batches", 20): break

    # 4. Final Aggregation
    if local_rank == 0:
        logger.info("Computing final routing metrics...")
        all_cvs, all_maxvios = [], []
        
        for mod_id, logits_list in hook_helper.layer_data.items():
            # [batch * num_batches, seq, num_experts]
            combined_logits = torch.cat(logits_list, dim=0) 
            probs = torch.softmax(combined_logits, dim=-1)
            usage_counts = probs.sum(dim=(0, 1)).numpy()
            target_usage = usage_counts.sum() / len(usage_counts)
            
            all_cvs.append(calculate_cv(usage_counts))
            all_maxvios.append(calculate_maxvio(usage_counts, target_usage))
            
        avg_cv, avg_maxvio = np.mean(all_cvs), np.mean(all_maxvios)
        logger.info(f"Turbo Results -> CV: {avg_cv:.8f}, Maxvio: {avg_maxvio:.8f}")
        
        tracker.add_metric("qwen3_turbo", "avg_cv", avg_cv)
        tracker.add_metric("qwen3_turbo", "avg_maxvio", avg_maxvio)
        tracker.save_day("qwen3_turbo")
        
        with open(output_dir / "turbo_report.txt", "w") as f:
            f.write(f"CV: {avg_cv:.8f}\nMaxvio: {avg_maxvio:.8f}\nStatus: {'PASS' if avg_cv < 0.001 else 'FAIL'}")

if __name__ == "__main__":
    main()
