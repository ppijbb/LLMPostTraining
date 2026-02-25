#!/bin/bash
# Script to run Qwen3-VL-30B-A3 Evaluation
# Usage: ./run_qwen3_eval.sh

set -e

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_train

# DeepSpeed/Distributed environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run evaluation
python -m torch.distributed.run --nproc_per_node=$WORLD_SIZE \
    eval/routing_benchmarks/qwen3_eval.py \
    --config eval/routing_benchmarks/config/qwen3_eval_config.yaml \
    --output_dir evaluation_results/qwen3_eval_$(date +%Y%m%d_%H%M%S)
