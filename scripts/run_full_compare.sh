#!/usr/bin/env bash
# 完整多模型对比实验（后台运行 eval）
# 用法：WORK_ROOT=/path/to/data GPU_ID=0 ./scripts/run_full_compare.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/data}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$REPO_ROOT/outputs"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cd "$REPO_ROOT"
echo "[$(date)] 启动 eval，数据目录=$WORK_ROOT GPU=$GPU_ID"
echo "日志: $REPO_ROOT/outputs/run_compare.log"
setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" HF_ENDPOINT="$HF_ENDPOINT" PYTHONUNBUFFERED=1 \
  "$PYTHON" -u run.py eval \
  --work-root "$WORK_ROOT" \
  --device "cuda:0" \
  >> "$REPO_ROOT/outputs/run_compare.log" 2>&1 < /dev/null &
echo "PID=$!  查看进度: tail -f $REPO_ROOT/outputs/run_compare.log"
