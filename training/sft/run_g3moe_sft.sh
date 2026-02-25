#!/bin/bash

# G3MoE SFT Training Script
# Usage: ./run_g3moe_sft.sh [config_file]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}      G3MoE SFT Training Script        ${NC}"
echo -e "${GREEN}========================================${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default config file
CONFIG_FILE="${1:-$SCRIPT_DIR/config/g3moe_training_config.json}"

echo -e "${YELLOW}Project Root:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}Config File:${NC} $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found at $CONFIG_FILE${NC}"
    echo "Available config files:"
    find "$SCRIPT_DIR/config" -name "*.json" -type f 2>/dev/null | sort || echo "  No config files found"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

# Create output directory
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['training_config']['output_dir'])")
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Output Directory:${NC} $OUTPUT_DIR"
echo -e "${YELLOW}CUDA Devices:${NC} $CUDA_VISIBLE_DEVICES"

# Check if required packages are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
for package in torch transformers trl peft datasets wandb; do
    if ! python3 -c "import $package" 2>/dev/null; then
        echo -e "${RED}Error: $package is not installed${NC}"
        echo "Please install required packages:"
        echo "pip install torch transformers trl peft datasets wandb"
        exit 1
    fi
done
echo -e "${GREEN}All dependencies found!${NC}"

# Check GPU availability
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo -e "${GREEN}GPU detected: $GPU_COUNT device(s) available${NC}"
else
    echo -e "${YELLOW}Warning: No GPU detected. Training will be slow.${NC}"
fi

# Start training
echo -e "${GREEN}Starting G3MoE SFT training...${NC}"
echo "Command: python3 $SCRIPT_DIR/custom_model_sft.py --config $CONFIG_FILE"

cd "$PROJECT_ROOT"

# Run training with error handling
if python3 "$SCRIPT_DIR/custom_model_sft.py" --config "$CONFIG_FILE"; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}Model saved to: $OUTPUT_DIR${NC}"
else
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi

# Optional: Run model evaluation
if [ "$2" = "--eval" ]; then
    echo -e "${YELLOW}Running model evaluation...${NC}"
    python3 -c "
import torch
from transformers import AutoTokenizer
from models.g3moe_model import G3MoEForCausalLM

print('Loading model for evaluation...')
tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_DIR')
model = G3MoEForCausalLM.from_pretrained('$OUTPUT_DIR')

print('Model loaded successfully!')
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

# Simple generation test
test_prompt = 'Hello, how are you?'
inputs = tokenizer(test_prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, do_sample=True, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'Test generation: {generated_text}')
"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}           Training Complete!           ${NC}"
echo -e "${GREEN}========================================${NC}" 