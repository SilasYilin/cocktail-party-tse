#!/usr/bin/env bash
# 解压 AISHELL 各说话人 tar.gz（若 download_datasets.sh 未完全解压）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/data}"
WAV_DIR="${WORK_ROOT}/raw/chinese/aishell/data_aishell/wav"

cd "$WAV_DIR"
count=0
total=$(ls -1 S*.tar.gz 2>/dev/null | wc -l)
for f in S*.tar.gz; do
  [[ -f "$f" ]] || continue
  spk="${f%.tar.gz}"
  if [[ -d "train/$spk" ]] && [[ $(find "train/$spk" -name '*.wav' 2>/dev/null | head -1 | wc -l) -gt 0 ]]; then
    continue
  fi
  tar -xzf "$f"
  rm -f "$f"
  count=$((count + 1))
  if (( count % 20 == 0 )); then
    echo "extracted $count / $total ..."
  fi
done
echo "done. wav files: $(find train -name '*.wav' 2>/dev/null | wc -l)"
