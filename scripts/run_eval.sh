#!/usr/bin/env bash
# separate + eval（需已生成混合数据与训练权重）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/data}"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "=== separate (test set) ===" >&2
"$PYTHON" run.py separate --work-root "$WORK_ROOT"
echo "=== eval (5-way TSE comparison) ===" >&2
"$PYTHON" run.py eval --work-root "$WORK_ROOT"
echo "=== done ===" >&2
