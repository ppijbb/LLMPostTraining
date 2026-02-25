#!/bin/bash
# Setup script for downloading baseline models
# This script downloads and caches baseline models for comparison

set -e

echo "=================================="
echo "SPECTRA Baseline Setup"
echo "=================================="
echo ""

# Configuration
CACHE_DIR="${CACHE_DIR:-./baselines}"
HF_TOKEN_FILE="$HOME/.huggingface/token"

echo "Cache directory: $CACHE_DIR"
echo ""

# Create cache directory
mkdir -p "$CACHE_DIR"

# Check for HuggingFace token (needed for gated models like LLaMA)
if [ -f "$HF_TOKEN_FILE" ]; then
    echo "✓ HuggingFace token found"
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
else
    echo "⚠ HuggingFace token not found at $HF_TOKEN_FILE"
    echo "  For gated models (LLaMA-3), you'll need to:"
    echo "  1. Create a token at https://huggingface.co/settings/tokens"
    echo "  2. Run: huggingface-cli login"
    echo ""
fi

# Function to download a model
download_model() {
    local model_name=$1
    local model_path=$2
    local target_dir="$CACHE_DIR/$model_name"
    
    echo "Downloading $model_name..."
    echo "  Model: $model_path"
    echo "  Target: $target_dir"
    
    if [ -d "$target_dir" ] && [ "$(ls -A $target_dir)" ]; then
        echo "  ✓ Already exists, skipping"
    else
        mkdir -p "$target_dir"
        
        # Use huggingface_hub to download
        python3 << EOF
from huggingface_hub import snapshot_download
import os

try:
    token = os.environ.get('HF_TOKEN')
    snapshot_download(
        repo_id="$model_path",
        local_dir="$target_dir",
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    print("  ✓ Downloaded successfully")
except Exception as e:
    print(f"  ✗ Download failed: {e}")
    exit(1)
EOF
    fi
    echo ""
}

# Download baseline models
echo "Starting baseline model downloads..."
echo ""

# Mixtral 8x7B (default comparison baseline)
download_model "mixtral" "mistralai/Mixtral-8x7B-v0.1"

# LLaMA-3 8B (alternative dense baseline)
download_model "llama3" "meta-llama/Meta-Llama-3-8B"

# Optional: Uncomment to download additional baselines
# download_model "deepseek_v3" "deepseek-ai/DeepSeek-V3"
# download_model "qwen25_moe" "Qwen/Qwen2.5-MoE-A14B"

echo "=================================="
echo "Baseline Setup Complete!"
echo "=================================="
echo ""
echo "Downloaded models are cached in: $CACHE_DIR"
echo ""
echo "Next steps:"
echo "  1. Update config/evaluation_config.yaml with your SPECTRA checkpoint path"
echo "  2. Run: ./scripts/run_all.sh"
echo ""

