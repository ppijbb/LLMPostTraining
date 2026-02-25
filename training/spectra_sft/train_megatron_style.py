#!/usr/bin/env python3
"""
SPECTRA SFT Training Script - Full Megatron-LM Trainer Structure
Based on NVIDIA Megatron-LM framework trainer architecture
"""

import os
import sys
import torch
import torch.distributed as dist
import argparse
import logging
from functools import partial
from typing import Dict, Any, Optional, Tuple

# Add parent directory to path
# Expect PYTHONPATH to include repo root

# Megatron-LM imports
try:
    from megatron import get_args, get_tokenizer, initialize_megatron, print_rank_0
    from megatron.core import mpu, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.model import GPTModel, GPTModelPipe
    from megatron.training import pretrain, train_step, forward_step
    from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
    from megatron.optimizer import get_megatron_optimizer
    from megatron.learning_rates import AnnealingLR
    from megatron.checkpointing import load_checkpoint, save_checkpoint
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False
    print("Megatron-LM not available, using fallback implementation")

from models import SPECTRAForCausalLM, SPECTRAConfig
from core.training_utils import load_config, setup_logging
from data.multi_domain_sft_dataset import create_multi_domain_dataset


def get_model_provider():
    """Megatron-LM style model provider function"""
    def model_provider(pre_process=True, post_process=True):
        """Build and return the model."""
        args = get_args()
        
        if MEGATRON_AVAILABLE:
            # Real Megatron-LM model
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
        else:
            # Fallback SPECTRA model
            config = load_config(args.config)
            spectra_config = SPECTRAConfig(**config.model_config)
            model = SPECTRAForCausalLM(spectra_config)
        
        return model
    
    return model_provider


def get_data_iterator():
    """Megatron-LM style data iterator provider"""
    args = get_args()
    
    if MEGATRON_AVAILABLE:
        # Real Megatron-LM data iterator
        from megatron.data.gpt_dataset import build_train_valid_test_datasets
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=[
                args.train_samples,
                args.valid_samples,
                args.test_samples
            ],
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=args.skip_warmup
        )
        
        from megatron.data.samplers import DistributedBatchSampler
        train_sampler = DistributedBatchSampler(
            train_ds,
            batch_size=args.micro_batch_size,
            rank=mpu.get_data_parallel_rank(),
            world_size=mpu.get_data_parallel_world_size(),
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size()
        )
        
        train_data_iterator = iter(
            torch.utils.data.DataLoader(
                train_ds,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
        )
    else:
        # Fallback data iterator
        config = load_config(args.config)
        dataset = create_multi_domain_dataset(config.data_config)
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )
        
        train_data_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=args.micro_batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
        )
    
    return train_data_iterator


def forward_step_func(data_iterator, model):
    """Megatron-LM style forward step function"""
    args = get_args()
    
    def loss_func(labels, output_tensor):
        """Loss function for forward step"""
        if MEGATRON_AVAILABLE:
            # Real Megatron-LM loss
            from megatron.model.language_model import parallel_lm_logits
            logits = parallel_lm_logits(
                output_tensor,
                unwrap_model(model).word_embeddings.weight,
                args.parallel_output
            )
            
            from megatron.model.utils import init_method_normal, scaled_init_method_normal
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                label_smoothing=args.label_smoothing
            )
        else:
            # Fallback loss
            loss = output_tensor.loss
        
        return loss, {'lm_loss': loss}
    
    return loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Megatron-LM style dataset provider"""
    args = get_args()
    
    if MEGATRON_AVAILABLE:
        # Real Megatron-LM datasets
        from megatron.data.gpt_dataset import build_train_valid_test_datasets
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=args.skip_warmup
        )
        return train_ds, valid_ds, test_ds
    else:
        # Fallback datasets
        config = load_config(args.config)
        dataset = create_multi_domain_dataset(config.data_config)
        return dataset, None, None


def pretrain_forward_step_func(data_iterator, model):
    """Megatron-LM style pretrain forward step"""
    args = get_args()
    
    # Get the batch
    if MEGATRON_AVAILABLE:
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    else:
        batch = next(data_iterator)
        tokens = batch['input_ids']
        labels = batch['labels']
        loss_mask = batch.get('attention_mask', None)
        attention_mask = batch.get('attention_mask', None)
        position_ids = None
    
    # Forward pass
    if MEGATRON_AVAILABLE:
        output_tensor = model(tokens, position_ids, attention_mask)
    else:
        outputs = model(input_ids=tokens, attention_mask=attention_mask, labels=labels)
        output_tensor = outputs.logits
        model_outputs = outputs  # Store for loss function
    
    # Loss function
    def loss_func(labels, output_tensor):
        if MEGATRON_AVAILABLE:
            from megatron.model.language_model import parallel_lm_logits
            logits = parallel_lm_logits(
                output_tensor,
                unwrap_model(model).word_embeddings.weight,
                args.parallel_output
            )
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                label_smoothing=args.label_smoothing
            )
        else:
            loss = model_outputs.loss
        
        return loss, {'lm_loss': loss}
    
    return output_tensor, partial(loss_func, labels)


def get_batch(data_iterator):
    """Megatron-LM style get_batch function"""
    args = get_args()
    
    if MEGATRON_AVAILABLE:
        # Real Megatron-LM get_batch
        from megatron.data.data_samplers import get_batch
        return get_batch(data_iterator)
    else:
        # Fallback get_batch
        batch = next(data_iterator)
        tokens = batch['input_ids']
        labels = batch['labels']
        loss_mask = batch.get('attention_mask', None)
        attention_mask = batch.get('attention_mask', None)
        
        # Generate position_ids if needed
        seq_length = tokens.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)
        
        return tokens, labels, loss_mask, attention_mask, position_ids


def get_args():
    """Megatron-LM style argument parsing"""
    parser = argparse.ArgumentParser(description='SPECTRA Megatron-LM Training')
    
    # Config
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    
    # Model arguments
    parser.add_argument('--num_layers', type=int, default=24)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--num_attention_heads', type=int, default=16)
    parser.add_argument('--seq_length', type=int, default=2048)
    parser.add_argument('--max_position_embeddings', type=int, default=2048)
    
    # Training arguments
    parser.add_argument('--micro_batch_size', type=int, default=4)
    parser.add_argument('--global_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--train_iters', type=int, default=1000)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    
    # Distributed
    parser.add_argument('--tensor_model_parallel_size', type=int, default=1)
    parser.add_argument('--pipeline_model_parallel_size', type=int, default=1)
    parser.add_argument('--data_parallel_size', type=int, default=1)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    
    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--data_impl', type=str, default='mmap')
    parser.add_argument('--split', type=str, default='949,50,1')
    parser.add_argument('--train_samples', type=int, default=None)
    parser.add_argument('--valid_samples', type=int, default=None)
    parser.add_argument('--test_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_warmup', action='store_true')
    parser.add_argument('--parallel_output', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        args.config = load_config(args.config)
    
    return args


def main():
    """Main Megatron-LM style pretrain function"""
    if MEGATRON_AVAILABLE:
        # Real Megatron-LM pretrain
        pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step_func,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
        )
    else:
        # Fallback pretrain
        args = get_args()
        
        # Initialize distributed
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12345'
            dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
            torch.cuda.set_device(args.rank)
        
        # Setup logging
        if hasattr(args, 'config') and args.config:
            setup_logging(args.config, args.rank)
        
        # Build model
        model_provider_func = get_model_provider()
        model = model_provider_func(pre_process=True, post_process=True)
        model = model.cuda()
        
        # Build optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Build data iterator
        train_data_iterator = get_data_iterator()
        
        # Training loop
        model.train()
        for iteration in range(args.train_iters):
            try:
                # Forward step
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(train_data_iterator)
                
                outputs = model(input_ids=tokens, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Backward step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                
                if args.rank == 0 and iteration % args.log_interval == 0:
                    print(f"Iteration {iteration}, Loss: {loss.item():.4f}")
            except StopIteration:
                # Restart data iterator
                train_data_iterator = get_data_iterator()
                continue


if __name__ == "__main__":
    main()