#!/usr/bin/env bash
# mix 已完成时：train -> separate -> eval
# 用法：GPU_ID=0 WORK_ROOT=/path/to/data ./scripts/run_train_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/data}"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs}"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTORCH_DISABLE_DYNAMO="${PYTORCH_DISABLE_DYNAMO:-1}"
mkdir -p "$LOG_DIR"

echo "=== train ===" | tee "$LOG_DIR/run_train.log"
"$PYTHON" run.py train --work-root "$WORK_ROOT" 2>&1 | tee -a "$LOG_DIR/run_train.log"

echo "=== separate ===" | tee "$LOG_DIR/run_separate.log"
"$PYTHON" run.py separate --work-root "$WORK_ROOT" 2>&1 | tee -a "$LOG_DIR/run_separate.log"

echo "=== eval ===" | tee "$LOG_DIR/run_eval.log"
"$PYTHON" run.py eval --work-root "$WORK_ROOT" 2>&1 | tee -a "$LOG_DIR/run_eval.log"

echo "=== done ===" | tee -a "$LOG_DIR/run_pipeline.log"
