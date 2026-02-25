#!/bin/bash

# G3MoE SFT Training Script with DeepSpeed Support
# Usage: ./run_g3moe_deepspeed.sh [config_file] [num_gpus]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}        Multi Modal G3MoE SFT DeepSpeed Training with 120K Context Length       ${NC}"
echo -e "${GREEN}================================================================================${NC}"

# Config file name
CONFIG_FILE_NAME="g3moe_deepspeed_config.json"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default config file and GPU count
CONFIG_FILE="${1:-$SCRIPT_DIR/config/$CONFIG_FILE_NAME}"

# Detect available GPUs if not specified
if [ -n "$2" ]; then
    NUM_GPUS="$2"
else
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        echo -e "${YELLOW}Warning:${NC} nvidia-smi not found, defaulting to $2 GPUs"
        NUM_GPUS=4
    fi
fi

NUM_GPUS=2 # fixed gpu count

echo -e "${YELLOW}Project Root:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}Config File:${NC} $CONFIG_FILE"
echo -e "${YELLOW}Number of GPUs:${NC} $NUM_GPUS"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found at $CONFIG_FILE${NC}"
    echo "Available configs:"
    find "$SCRIPT_DIR/config" -name "*.json" -type f 2>/dev/null | sort || echo "  No config files found"
    exit 1
fi

# Check if DeepSpeed config is specified and exists
DEEPSPEED_CONFIG=$(python3 -c "import json; config=json.load(open('$CONFIG_FILE')); print(config['model_config'].get('deepspeed_config', ''))" 2>/dev/null || echo "")
if [ -n "$DEEPSPEED_CONFIG" ] && [ "$DEEPSPEED_CONFIG" != "null" ] && [ "$DEEPSPEED_CONFIG" != "None" ]; then
    if [ ! -f "$DEEPSPEED_CONFIG" ]; then
        echo -e "${RED}Error: DeepSpeed config not found at $DEEPSPEED_CONFIG${NC}"
        exit 1
    fi
    echo -e "${BLUE}DeepSpeed Config:${NC} $DEEPSPEED_CONFIG"
fi

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPUS-1)) | paste -sd, -)
fi
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# CUDA allocator tuning to reduce fragmentation and OOM risk
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True
# Avoid heavy sync/blocking debug flags which increase memory/stalls
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export FLASH_ATTENTION_2_ENABLED=true
# if system could use flash attention 3, use the following variables
export FLASH_ATTENTION_DETERMINISTIC=1
export FLASH_ATTENTION_SKIP_ALIBI=1
export TORCH_CUDNN_V8_API_ENABLED=1

export OMP_NUM_THREADS=$(nproc)  # 논리 코어 전체

export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=WARN
export DEEPSPEED_LOG_LEVEL=DEBUG
export WANDB_LOG_MODEL=end

export TORCH_DISTRIBUTED_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1 
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_DEBUG=INFO
export NCCL_PROTO=LL
export NCCL_ALGO=Ring
export DEEPSPEED_AUTOTP=1
export DS_AUTOTP=1
export DEEPSPEED_ENABLE_TP=1
export DS_ENABLE_TP=1
export ACCELERATE_USE_DEEPSPEED=1
export ACCELERATE_DISTRIBUTED_TYPE=DEEPSPEED
export ACCELERATE_DISABLE_RICH=0
export TORCH_CUDA_ARCH_LIST="8.0"
export SAFETENSORS_FAST_GPU=1
export HF_ENABLE_PARALLEL_LOADING=1
export HF_PARALLEL_LOADING_WORKERS=$NUM_GPUS
export HF_DATASETS_OFFLINE=0
export PROFILE_TRAINING=0
export HF_DATASETS_CACHE="/mls/conan/datasets_cache"
# Only log from rank 0; avoid disabling W&B globally
unset WANDB_DISABLED_IN_SUBPROCESS 2>/dev/null || true
# Create output directory
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training_config']['output_dir'])")
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Output Directory:${NC} $OUTPUT_DIR"
echo -e "${YELLOW}CUDA Devices:${NC} $CUDA_VISIBLE_DEVICES"

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
for package in torch transformers trl peft datasets wandb deepspeed; do
    if ! python -c "import $package" 2>/dev/null; then
        echo -e "${RED}Error: $package is not installed${NC}"
        python -c "import $package" 2>&1 | sed 's/^/    /'
        echo "$package is not installed"
        echo "Please install required packages:"
        echo "pip install torch transformers trl peft datasets wandb deepspeed"
        exit 1
    fi
done
echo -e "${GREEN}All dependencies found!${NC}"

# Check GPU availability and memory
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo -e "${GREEN}GPU detected: $GPU_COUNT device(s) available${NC}"
    
    # Check GPU memory for each device
    python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / (1024**3)
    print(f'GPU {i}: {props.name}, {memory_gb:.1f}GB VRAM')
    if memory_gb < 40:
        print(f'WARNING: GPU {i} has less than 40GB VRAM. 120K context may cause OOM.')
"
else
    echo -e "${RED}Error: No GPU detected. DeepSpeed requires CUDA.${NC}"
    exit 1
fi

# Check system memory for DeepSpeed CPU offload
TOTAL_RAM_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
echo -e "${YELLOW}System RAM:${NC} ${TOTAL_RAM_GB}GB"
if [ "$TOTAL_RAM_GB" -lt 64 ]; then
    echo -e "${YELLOW}Warning: Less than 64GB RAM. DeepSpeed CPU offload may be slow.${NC}"
fi

# Determine the training command based on number of GPUs
cd "$PROJECT_ROOT"

# To run with 1 GPU, use the following command:
# python3 $SCRIPT_DIR/custom_model_sft.py --config $CONFIG_FILE

# To run with multiple GPUs, use the following command:
# torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_DIR/custom_model_sft.py --config $CONFIG_FILE


echo -e "${GREEN}Starting DeepSpeed training with $NUM_GPUS GPUs...${NC}"
TRAIN_CMD="accelerate launch \
    --config_file $SCRIPT_DIR/config/accelerate.yaml \
        $SCRIPT_DIR/custom_model_sft.py --config $CONFIG_FILE"

# Run training with error handling
if eval "$TRAIN_CMD"; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}Model saved to: $OUTPUT_DIR${NC}"
    
    # Show model size info
    echo -e "${BLUE}Model information:${NC}"
    python3 -c "
import os
import torch
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
if (output_dir / 'pytorch_model.bin').exists():
    model_file = output_dir / 'pytorch_model.bin'
    size_mb = model_file.stat().st_size / (1024*1024)
    print(f'Model file size: {size_mb:.1f} MB')
elif (output_dir / 'adapter_model.safetensors').exists():
    adapter_file = output_dir / 'adapter_model.safetensors'
    size_mb = adapter_file.stat().st_size / (1024*1024)
    print(f'LoRA adapter size: {size_mb:.1f} MB')
"
else
    echo -e "${RED}Training failed!${NC}"
    echo -e "${YELLOW}Troubleshooting tips for 120K context:${NC}"
    echo "1. Try reducing batch size: per_device_train_batch_size=1"
    echo "2. Increase gradient accumulation: gradient_accumulation_steps=64"
    echo "3. Use DeepSpeed ZeRO-3 with CPU offload"
    echo "4. Enable gradient checkpointing"
    echo "5. Reduce LoRA rank if using LoRA"
    exit 1
fi

# Memory usage summary
echo -e "${BLUE}Final GPU memory usage:${NC}"
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        cached = torch.cuda.memory_reserved(i) / (1024**3)
        print(f'GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached')
"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}      DeepSpeed Training Complete!     ${NC}"
echo -e "${GREEN}========================================${NC}" 
rm -rf $OUTPUT_DIR 