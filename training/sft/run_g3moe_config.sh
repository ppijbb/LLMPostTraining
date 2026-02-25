#!/bin/bash

# G3MoE SFT Training Script with Config File
# Usage: ./run_g3moe_config.sh [config_file] [options]
# Options:
#   --resume: Resume from checkpoint
#   --dry-run: Show config and exit without training
#   --multi-gpu: Use multiple GPUs with torchrun
#   --help: Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Function to show help
show_help() {
    echo "G3MoE SFT Training Script"
    echo "========================="
    echo ""
    echo "Usage: $0 [config_file] [options]"
    echo ""
    echo "Arguments:"
    echo "  config_file    Path to JSON config file (default: config/sft/g3moe_training_config.json)"
    echo ""
    echo "Options:"
    echo "  --resume       Resume training from checkpoint"
    echo "  --dry-run      Show configuration and exit without training"
    echo "  --multi-gpu    Use multiple GPUs with torchrun"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                                    # Use default config"
    echo "  $0 config/sft/g3moe_deepspeed_config.json           # Use specific config"
    echo "  $0 --multi-gpu                                       # Multi-GPU training"
    echo "  $0 my_config.json --resume                           # Resume training"
    echo "  $0 --dry-run                                         # Check config only"
    echo ""
    echo "Available config files:"
    find config/sft -name "*.json" -type f 2>/dev/null | sort || echo "  No config files found in config/sft/"
}

# Parse arguments
CONFIG_FILE=""
RESUME=false
DRY_RUN=false
MULTI_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --multi-gpu)
            MULTI_GPU=true
            shift
            ;;
        --*)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"
            else
                echo "❌ Multiple config files specified"
                exit 1
            fi
            shift
            ;;
    esac
done

# Set default config file if not specified
CONFIG_FILE=${CONFIG_FILE:-"config/sft/g3moe_training_config.json"}

echo "🚀 G3MoE SFT Training Script"
echo "============================"
echo "Config file: $CONFIG_FILE"
echo "Resume: $RESUME"
echo "Dry run: $DRY_RUN"
echo "Multi-GPU: $MULTI_GPU"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available config files:"
    find config/sft -name "*.json" -type f 2>/dev/null | sort || echo "  No config files found in config/sft/"
    exit 1
fi

# Validate JSON config
echo "🔍 Validating config file..."
if ! python -c "import json; json.load(open('$CONFIG_FILE'))" 2>/dev/null; then
    echo "❌ Invalid JSON in config file: $CONFIG_FILE"
    exit 1
fi
echo "✅ Config file is valid JSON"

# Show config summary
echo ""
echo "📋 Configuration Summary:"
echo "------------------------"
python -c "
import json
config = json.load(open('$CONFIG_FILE'))
print(f\"Model: {config['model_config']['model_name_or_path']}\")
print(f\"Dataset: {config['data_config']['dataset_name']}\")
print(f\"Max seq length: {config['data_config']['max_seq_length']:,}\")
print(f\"Epochs: {config['training_config']['num_train_epochs']}\")
print(f\"Batch size: {config['training_config']['per_device_train_batch_size']}\")
print(f\"Learning rate: {config['training_config']['learning_rate']}\")
print(f\"Output dir: {config['training_config']['output_dir']}\")
if config['model_config'].get('deepspeed_config'):
    print(f\"DeepSpeed: {config['model_config']['deepspeed_config']}\")
print(f\"LoRA: {config['model_config']['use_lora']} (r={config['model_config']['lora_r']})\")
g3moe = config['model_config']['g3moe_params']
print(f\"G3MoE: {g3moe['n_routed_experts']} experts, {g3moe['n_group']} groups\")
"

# Exit if dry run
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🏁 Dry run completed. Exiting without training."
    exit 0
fi

# Check Python environment
echo ""
echo "🔍 Checking Python environment..."
python --version

# Check required packages
REQUIRED_PACKAGES=("transformers" "trl" "peft" "torch" "datasets")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "$package" > /dev/null 2>&1; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo "❌ Missing required packages: ${MISSING_PACKAGES[*]}"
    echo "Please install requirements:"
    echo "pip install -r requirements.txt"
    exit 1
fi
echo "✅ All required packages are installed"

# Check GPU availability and memory
echo ""
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "🎮 GPU Status ($GPU_COUNT GPUs detected):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | while read line; do
        echo "  GPU $line"
    done
    
    # Check if multi-GPU is requested but only one GPU available
    if [ "$MULTI_GPU" = true ] && [ "$GPU_COUNT" -eq 1 ]; then
        echo "⚠️  Multi-GPU requested but only 1 GPU detected. Using single GPU."
        MULTI_GPU=false
    fi
    
    # Memory check
    MIN_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$MIN_MEMORY" -lt 10000 ]; then
        echo "⚠️  Low GPU memory detected (${MIN_MEMORY}MB free). Consider using DeepSpeed or reducing batch size."
    fi
else
    echo "⚠️  No GPU detected. Training will be very slow on CPU."
    MULTI_GPU=false
fi

# Check disk space
echo ""
echo "💾 Checking disk space..."
OUTPUT_DIR=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['training_config']['output_dir'])")
AVAILABLE_SPACE=$(df -BG "$(dirname "$OUTPUT_DIR")" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    echo "⚠️  Low disk space: ${AVAILABLE_SPACE}GB available. Model checkpoints may require significant space."
else
    echo "✅ Sufficient disk space: ${AVAILABLE_SPACE}GB available"
fi

# Set environment variables for optimization
echo ""
echo "⚙️  Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=g3moe-sft
export HF_HOME=~/.cache/huggingface

# Additional optimizations
if [ "$MULTI_GPU" = true ]; then
    export NCCL_DEBUG=INFO
    export CUDA_LAUNCH_BLOCKING=0
fi

echo "✅ Environment configured"

# Prepare training command
TRAIN_CMD="python sft/custom_model_sft.py --config $CONFIG_FILE"

# Add resume flag if requested
if [ "$RESUME" = true ]; then
    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | wc -l)" -gt 0 ]; then
        LATEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/checkpoint-* | head -1)
        echo "📂 Found checkpoint: $LATEST_CHECKPOINT"
        # Note: TRL handles resume automatically if output_dir contains checkpoints
    else
        echo "⚠️  No checkpoints found in $OUTPUT_DIR. Starting fresh training."
    fi
fi

# Setup multi-GPU training
if [ "$MULTI_GPU" = true ]; then
    echo ""
    echo "🔄 Setting up multi-GPU training with torchrun..."
    TRAIN_CMD="torchrun --nproc_per_node=$GPU_COUNT --master_port=29500 sft/custom_model_sft.py --config $CONFIG_FILE"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save training info
echo ""
echo "📝 Saving training info..."
cat > "$OUTPUT_DIR/training_info.txt" << EOF
G3MoE SFT Training Information
=============================
Start time: $(date)
Config file: $CONFIG_FILE
Resume: $RESUME
Multi-GPU: $MULTI_GPU ($GPU_COUNT GPUs)
Command: $TRAIN_CMD

System Info:
- Python: $(python --version)
- PyTorch: $(python -c "import torch; print(torch.__version__)")
- Transformers: $(python -c "import transformers; print(transformers.__version__)")
- TRL: $(python -c "import trl; print(trl.__version__)")
- CUDA: $(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'Not available')")

GPU Info:
$(nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "No GPU info available")
EOF

# Start training with error handling
echo ""
echo "🏃 Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Trap to handle interruption
trap 'echo ""; echo "⚠️  Training interrupted. Checkpoints saved in: $OUTPUT_DIR"; exit 130' INT

# Run training and capture exit code
set +e
$TRAIN_CMD
TRAIN_EXIT_CODE=$?
set -e

# Training completion
echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Results saved in: $OUTPUT_DIR"
    echo "📊 Check logs and model files in the output directory"
    
    # Show final model info
    if [ -f "$OUTPUT_DIR/pytorch_model.bin" ] || [ -f "$OUTPUT_DIR/model.safetensors" ]; then
        echo "🎯 Final model saved successfully"
    fi
    
    # Show training summary
    if [ -f "$OUTPUT_DIR/trainer_state.json" ]; then
        echo "📈 Training summary available in trainer_state.json"
    fi
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "📁 Check logs in: $OUTPUT_DIR"
    echo "💡 Common issues:"
    echo "   - Out of memory: Reduce batch size or use DeepSpeed"
    echo "   - CUDA errors: Check GPU compatibility"
    echo "   - Dataset errors: Verify dataset name and access"
    exit $TRAIN_EXIT_CODE
fi

# Final cleanup and summary
echo ""
echo "🏁 Training session completed at $(date)"
echo "📋 Session summary:"
echo "   - Config: $CONFIG_FILE"
echo "   - Output: $OUTPUT_DIR"
echo "   - Duration: Training completed"
echo "   - Status: $([ $TRAIN_EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED")" 