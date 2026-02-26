#!/bin/bash
# Convenience wrapper for running the full Seqorth evaluation pipeline

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo "=================================="
echo "Seqorth 7-Day Evaluation Pipeline"
echo "=================================="
echo ""

# Configuration
CONFIG="${CONFIG:-$SCRIPT_DIR/../config/evaluation_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./evaluation_results}"
PYTHON="${PYTHON:-python3}"

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Output directory: $OUTPUT_DIR"
echo "  Python: $PYTHON"
echo ""

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    echo ""
    echo "Please create config/evaluation_config.yaml first."
    echo "See config/evaluation_config.yaml for template."
    exit 1
fi

# Check required settings
echo "Validating configuration..."
python3 << EOF
import yaml
import sys

with open("$CONFIG", 'r') as f:
    config = yaml.safe_load(f)

errors = []

# Check checkpoint path
checkpoint = config.get('model', {}).get('checkpoint_path')
if not checkpoint or checkpoint == "/path/to/seqorth/checkpoint":
    errors.append("  - model.checkpoint_path must be set to your Seqorth checkpoint")

# Check WandB run_id
run_id = config.get('wandb', {}).get('run_id')
if not run_id or run_id == "bzrw39zy":
    errors.append("  - wandb.run_id should be updated to your actual run ID")

if errors:
    print("⚠ Configuration warnings:")
    for error in errors:
        print(error)
    print("")
    print("You can continue, but you may need to update these values.")
    print("")
else:
    print("✓ Configuration looks good")
    print("")

EOF

# Prompt user to continue
read -p "Start evaluation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Starting evaluation pipeline..."
echo ""

# Run the pipeline
cd "$PROJECT_ROOT"

$PYTHON eval/routing_benchmarks/run_full_pipeline.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    "$@"

# Check exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=================================="
    echo "✓ Pipeline completed successfully!"
    echo "=================================="
    echo ""
    echo "Results are available in: $OUTPUT_DIR"
    echo ""
    echo "Key outputs:"
    echo "  - Training curves: $OUTPUT_DIR/day0/"
    echo "  - Benchmark results: $OUTPUT_DIR/day1_2/"
    echo "  - Expert analysis: $OUTPUT_DIR/day3_4/"
    echo "  - Efficiency metrics: $OUTPUT_DIR/day5/"
    echo "  - LaTeX tables: $OUTPUT_DIR/day6/"
    echo ""
    echo "Next steps:"
    echo "  1. Review $OUTPUT_DIR/day0/sanity_check_report.txt"
    echo "  2. Check LaTeX tables in $OUTPUT_DIR/day6/"
    echo "  3. Include figures and tables in your paper"
    echo ""
else
    echo "=================================="
    echo "✗ Pipeline failed (exit code: $EXIT_CODE)"
    echo "=================================="
    echo ""
    echo "Check the logs for details:"
    echo "  - Full log: $OUTPUT_DIR/full_pipeline_log.txt"
    echo "  - Individual day logs in each day's directory"
    echo ""
    echo "Common issues:"
    echo "  - CUDA OOM: Reduce batch_size_per_gpu in config"
    echo "  - Model loading: Check checkpoint_path"
    echo "  - WandB errors: Verify run_id or use --skip_wandb"
    echo ""
fi

exit $EXIT_CODE

