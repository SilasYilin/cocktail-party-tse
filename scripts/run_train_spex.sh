#!/usr/bin/env bash
# 后台训练 SpEx+ 对照模型
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/data}"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
LOG="${LOG:-$REPO_ROOT/outputs/run_train_spex.log}"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTORCH_DISABLE_DYNAMO="${PYTORCH_DISABLE_DYNAMO:-1}"
mkdir -p "$(dirname "$LOG")"

echo "=== train_spex $(date -Iseconds) ===" | tee "$LOG"
nohup "$PYTHON" run.py train_spex --work-root "$WORK_ROOT" >> "$LOG" 2>&1 &
echo "train_spex PID=$! (log: $LOG)"
