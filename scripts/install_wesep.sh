#!/usr/bin/env bash
# 安装 WeSep 及其推理依赖（PyPI 无 wesep 包）
set -euo pipefail

pip install "git+https://github.com/wenet-e2e/WeSep.git" silero-vad onnxruntime
echo "[done] WeSep installed. See docs/checkpoints.md for weight download."
