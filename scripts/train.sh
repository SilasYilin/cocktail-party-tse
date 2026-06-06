#!/usr/bin/env bash
# 训练 AttentionTSE
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/data}"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTORCH_DISABLE_DYNAMO="${PYTORCH_DISABLE_DYNAMO:-1}"

exec "$PYTHON" run.py train --work-root "$WORK_ROOT"
