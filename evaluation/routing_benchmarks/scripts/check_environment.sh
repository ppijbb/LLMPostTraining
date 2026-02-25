#!/bin/bash
# Environment checker for SPECTRA evaluation pipeline
# Verifies all dependencies and configurations are properly set up

echo "=================================="
echo "SPECTRA Evaluation Environment Check"
echo "=================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to check command
check_command() {
    local cmd=$1
    local package=$2
    
    if command -v $cmd &> /dev/null; then
        echo -e "${GREEN}✓${NC} $cmd found"
    else
        echo -e "${RED}✗${NC} $cmd not found (install: $package)"
        ERRORS=$((ERRORS+1))
    fi
}

# Function to check Python package
check_python_package() {
    local package=$1
    local import_name=${2:-$package}
    
    if python3 -c "import $import_name" 2>/dev/null; then
        # Get version
        version=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null)
        echo -e "${GREEN}✓${NC} $package ($version)"
    else
        echo -e "${RED}✗${NC} $package not installed (pip install $package)"
        ERRORS=$((ERRORS+1))
    fi
}

echo "System Information:"
echo "  OS: $(uname -s)"
echo "  Kernel: $(uname -r)"
echo "  Python: $(python3 --version)"
echo ""

echo "Checking Required Commands..."
check_command "python3" "python3"
check_command "git" "git"
check_command "nvcc" "cuda-toolkit" || echo -e "  ${YELLOW}⚠${NC} CUDA toolkit not found (optional but recommended)"
echo ""

echo "Checking Python Packages..."

# Core dependencies
check_python_package "torch"
check_python_package "transformers"
check_python_package "datasets"
check_python_package "accelerate"
check_python_package "yaml" "yaml"
check_python_package "numpy"
check_python_package "pandas"

# Evaluation tools
check_python_package "lm_eval" "lm_eval" || echo -e "  ${YELLOW}⚠${NC} lm-eval not found (pip install lm-eval)"

# Visualization
check_python_package "matplotlib"
check_python_package "seaborn"

# Optional but recommended
echo ""
echo "Optional Dependencies:"
if python3 -c "import vllm" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} vllm (for throughput measurement)"
else
    echo -e "${YELLOW}⚠${NC} vllm not installed (pip install vllm) - needed for day5"
    WARNINGS=$((WARNINGS+1))
fi

if python3 -c "import wandb" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} wandb (for training curve extraction)"
else
    echo -e "${YELLOW}⚠${NC} wandb not installed (pip install wandb) - needed for day0"
    WARNINGS=$((WARNINGS+1))
fi

echo ""
echo "Checking GPU Availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✓${NC} Found $GPU_COUNT GPU(s)"
    echo ""
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | nl -v 0 | sed 's/^/  GPU /'
else
    echo -e "${RED}✗${NC} nvidia-smi not found - no GPUs detected"
    ERRORS=$((ERRORS+1))
fi

echo ""
echo "Checking CUDA/PyTorch Configuration..."
python3 << 'EOF'
import torch
import sys

print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
else:
    print("  ⚠ CUDA not available - evaluation will be very slow!")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS+1))
fi

echo ""
echo "Checking Configuration Files..."

CONFIG_DIR="$(dirname "$0")/../config"

if [ -f "$CONFIG_DIR/evaluation_config.yaml" ]; then
    echo -e "${GREEN}✓${NC} evaluation_config.yaml found"
    
    # Check if it's been customized
    if grep -q "/path/to/spectra/checkpoint" "$CONFIG_DIR/evaluation_config.yaml"; then
        echo -e "  ${YELLOW}⚠${NC} checkpoint_path needs to be updated"
        WARNINGS=$((WARNINGS+1))
    fi
else
    echo -e "${RED}✗${NC} evaluation_config.yaml not found"
    ERRORS=$((ERRORS+1))
fi

if [ -f "$CONFIG_DIR/baseline_models.yaml" ]; then
    echo -e "${GREEN}✓${NC} baseline_models.yaml found"
else
    echo -e "${RED}✗${NC} baseline_models.yaml not found"
    ERRORS=$((ERRORS+1))
fi

echo ""
echo "Checking Disk Space..."
df -h . | tail -n 1 | awk '{
    size = $4;
    if (match(size, /^[0-9]+T/)) {
        print "  ✓ Available: " size " (sufficient)";
    } else if (match(size, /^[0-9]+G/)) {
        num = substr(size, 1, length(size)-1);
        if (num > 100) {
            print "  ✓ Available: " size " (sufficient)";
        } else {
            print "  ⚠ Available: " size " (may be tight for baselines)";
        }
    } else {
        print "  ✗ Available: " size " (insufficient)";
    }
}'

echo ""
echo "=================================="
echo "Summary"
echo "=================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Your environment is ready for SPECTRA evaluation."
    echo ""
    echo "Next steps:"
    echo "  1. Update config/evaluation_config.yaml with your checkpoint path"
    echo "  2. Run: ./scripts/setup_baselines.sh (to download baseline models)"
    echo "  3. Run: ./scripts/run_all.sh (to start evaluation)"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Checks passed with $WARNINGS warning(s)${NC}"
    echo ""
    echo "You can proceed, but some optional features may not work."
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before running the evaluation."
    exit 1
fi

