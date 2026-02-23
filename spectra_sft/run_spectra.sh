#!/bin/bash

# SPECTRA SFT Training Script with DeepSpeed Support
# Usage: ./run_spectra.sh [config_file] [num_gpus]
#        ./run_spectra.sh baseline [config_file] [num_gpus]  → SPECTRA 비활성화(Qwen3 원래 MoE), backward 에러 회피
#
# 참고: "tensor a (0) vs tensor b (2048)" backward 에러 시 원인 추적:
#       DETECT_ANOMALY=1 ./run_spectra.sh baseline  → 실패한 backward 연산 위치가 로그에 출력됨 (한 번만 실행 후 로그 확인).

set -e

# JIT cpp_extension: set before any Python runs so all archs aren't compiled (avoids hang/slow compile).
# Override with SPECTRA_CUDA_ARCH if needed (e.g. 9.0 for H100).
export TORCH_CUDA_ARCH_LIST="${SPECTRA_CUDA_ARCH:-8.0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# [baseline 모드] 첫 인자가 baseline 이면 SPECTRA 비활성화 (통과했을 때와 동일)
RUN_BASELINE=0
if [ "$1" = "baseline" ] || [ "$1" = "no-spectra" ]; then
    RUN_BASELINE=1
    export SPECTRA_DISABLE_ALL=1
    shift
    echo -e "${YELLOW}[baseline 모드] SPECTRA 비활성화 (Qwen3 원래 MoE만 사용)${NC}"
fi

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}        SPECTRA SFT DeepSpeed Training with Small Model Configuration      ${NC}"
echo -e "${GREEN}================================================================================${NC}"

# Config file name
CONFIG_FILE_NAME="spectra_qwen_config.json"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default config file and GPU count
CONFIG_FILE="${1:-$SCRIPT_DIR/config/$CONFIG_FILE_NAME}"

# Detect available GPUs if not specified (default: all visible GPUs)
if [ -n "$2" ]; then
    NUM_GPUS="$2"
else
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        echo -e "${YELLOW}Warning:${NC} nvidia-smi not found, defaulting to 2 GPUs"
        NUM_GPUS=2
    fi
fi

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
# Keep launcher world size and visible devices consistent.
# - If CUDA_VISIBLE_DEVICES is already set, honor it and optionally trim to NUM_GPUS.
# - If not set, create 0..NUM_GPUS-1.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -r -a _VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    if [ "${#_VISIBLE_GPU_IDS[@]}" -lt "$NUM_GPUS" ]; then
        NUM_GPUS=${#_VISIBLE_GPU_IDS[@]}
    fi
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${_VISIBLE_GPU_IDS[*]:0:$NUM_GPUS}")
    export CUDA_VISIBLE_DEVICES
else
    export CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPUS-1)) | paste -sd, -)
fi
# Recompute NUM_GPUS from final CUDA_VISIBLE_DEVICES for accelerate launch
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# CUDA allocator tuning to reduce fragmentation and OOM risk
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
# Avoid heavy sync/blocking debug flags which increase memory/stalls
export CUDA_LAUNCH_BLOCKING=0
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
# Enable DeepSpeed AutoTP when tensor_parallel.autotp_size > 1 in config (required to avoid OOM)
export DEEPSPEED_AUTOTP=1
export DS_AUTOTP=1
export DEEPSPEED_ENABLE_TP=1
export DS_ENABLE_TP=1
export ACCELERATE_USE_DEEPSPEED=1
export ACCELERATE_DISTRIBUTED_TYPE=DEEPSPEED
export ACCELERATE_DISABLE_RICH=0
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
echo -e "${BLUE}Routing KPI (CV-first): moe/avg_expert_cv, moe/cv_window_mean, moe/cv_window_std, moe/avg_maxvio${NC}"
echo -e "${BLUE}CV+Quality pass: moe/cv_quality_pass=1 when CV stable and quality defence (entropy floor, active_experts, unused_experts) not degraded${NC}"
if [ "$RUN_BASELINE" = "1" ]; then
    echo -e "${YELLOW}[baseline] SPECTRA_DISABLE_ALL=1 → Qwen3 원래 MoE만 사용 (backward 에러 회피)${NC}"
fi

# Check dependencies
# echo -e "${YELLOW}Checking dependencies...${NC}"
# echo -e "Python path: $(which python)"
# for package in torch transformers trl peft datasets wandb deepspeed; do
#     if ! python -c "import $package" 2>/dev/null; then
#         echo -e "${RED}Error: $package is not installed${NC}"
#         python -c "import $package" 2>&1 | sed 's/^/    /'
#         echo "$package is not installed"
#         echo "Please install required packages:"
#         echo "pip install torch transformers trl peft datasets wandb deepspeed"
#         exit 1
#     fi
# done
# echo -e "${GREEN}All dependencies found!${NC}"

# Check GPU availability and memory
# if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
#     GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
#     echo -e "${GREEN}GPU detected: $GPU_COUNT device(s) available${NC}"
#     
#     # Check GPU memory for each device
#     python3 -c "
# import torch
# for i in range(torch.cuda.device_count()):
#     props = torch.cuda.get_device_properties(i)
#     memory_gb = props.total_memory / (1024**3)
#     print(f'GPU {i}: {props.name}, {memory_gb:.1f}GB VRAM')
# "
# else
#     echo -e "${RED}Error: No GPU detected. DeepSpeed requires CUDA.${NC}"
#     exit 1
# fi

# Check system memory for DeepSpeed CPU offload
# TOTAL_RAM_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
# echo -e "${YELLOW}System RAM:${NC} ${TOTAL_RAM_GB}GB"
# if [ "$TOTAL_RAM_GB" -lt 32 ]; then
#     echo -e "${YELLOW}Warning: Less than 32GB RAM. DeepSpeed CPU offload may be slow.${NC}"
# fi

# Determine the training command based on number of GPUs
cd "$PROJECT_ROOT"

echo -e "${GREEN}Starting DeepSpeed training with $NUM_GPUS GPUs...${NC}"

# baseline 모드: 커맨드라인 --baseline 전달 (env는 accelerate 자식에 안 넘어가서 인자로 전달)
if [ "$RUN_BASELINE" = "1" ]; then
    TRAIN_CMD="accelerate launch \
    --config_file $SCRIPT_DIR/../spectra_sft/config/accelerate.yaml \
    --num_processes $NUM_GPUS \
        $SCRIPT_DIR/train_spectra.py --config $CONFIG_FILE --baseline"
else
    TRAIN_CMD="accelerate launch \
    --config_file $SCRIPT_DIR/../spectra_sft/config/accelerate.yaml \
    --num_processes $NUM_GPUS \
        $SCRIPT_DIR/train_spectra.py --config $CONFIG_FILE"
fi

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
    echo -e "${YELLOW}Troubleshooting tips:${NC}"
    echo "1. Try reducing batch size: per_device_train_batch_size=1"
    echo "2. Increase gradient accumulation: gradient_accumulation_steps=8"
    echo "3. Use ZeRO-3 without param offload, or increase tensor_parallel / reduce batch or seq length (param offload 금지)"
    echo "4. Enable gradient checkpointing"
    echo "5. Reduce model size if initializing from scratch"
    echo ""
    echo -e "${YELLOW}If Rank 1 hits 'Watchdog caught collective operation timeout' (ZeRO-3 all_gather):${NC}"
    echo "  → Rank 0 likely OOM'd first. Check Rank 0 logs for OutOfMemoryError."
    echo "  → ZeRO-3 only (no fallback): reduce stage3_max_live_parameters in deepspeed_zero3.json (e.g. 25e6)."
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

