#!/bin/bash
# Vision path만 ZeRO-2 + 다중 GPU로 빠르게 테스트 (학습과 동일 조건).
# 사용: ./run_test_vision.sh [num_gpus]
#       num_gpus 생략 시 4

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NUM_GPUS="${1:-4}"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"

echo "Running vision backward test (accelerate + DeepSpeed ZeRO-2, $NUM_GPUS GPUs)..."

# multi-GPU만 쓰는 config로 launch (DeepSpeed는 test_vision_backward.py 내부에서 초기화)
accelerate launch \
  --config_file "$SCRIPT_DIR/config/accelerate_vision_test.yaml" \
  --num_processes "$NUM_GPUS" \
  "$SCRIPT_DIR/test_vision_backward.py"

echo "Vision test finished."
