#!/usr/bin/env bash
# 扩数据后全流程：mix -> train -> separate -> eval
# 用法：GPU_ID=0 WORK_ROOT=/path/to/data ./scripts/run_pipeline.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/data}"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs}"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
mkdir -p "$LOG_DIR"

rm -f "$LOG_DIR/speaker_splits.json"
rm -f checkpoints/attention_tse/best.pt

echo "=== mix ===" | tee "$LOG_DIR/run_mix.log"
"$PYTHON" run.py mix --work-root "$WORK_ROOT" 2>&1 | tee -a "$LOG_DIR/run_mix.log"

echo "=== train ===" | tee "$LOG_DIR/run_train.log"
"$PYTHON" run.py train --work-root "$WORK_ROOT" 2>&1 | tee -a "$LOG_DIR/run_train.log"

echo "=== separate ===" | tee "$LOG_DIR/run_separate.log"
"$PYTHON" run.py separate --work-root "$WORK_ROOT" 2>&1 | tee -a "$LOG_DIR/run_separate.log"

echo "=== eval ===" | tee "$LOG_DIR/run_eval.log"
"$PYTHON" run.py eval --work-root "$WORK_ROOT" 2>&1 | tee -a "$LOG_DIR/run_eval.log"

echo "=== done ==="
