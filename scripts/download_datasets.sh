#!/usr/bin/env bash
# 下载并解压 LibriSpeech train-clean-100、AISHELL-1、MUSAN
# 用法：WORK_ROOT=/path/to/data ./scripts/download_datasets.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/data}"
DL="${WORK_ROOT}/downloads"
RAW="${WORK_ROOT}/raw"
mkdir -p "$DL" "$RAW/english/librispeech/LibriSpeech" "$RAW/chinese/aishell" "$RAW/noise"

download() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "[skip] exists: $out"
    return 0
  fi
  echo "[wget] $url -> $out"
  wget -c --tries=5 --timeout=60 -O "$out" "$url"
}

# --- LibriSpeech train-clean-100 ---
TC100_TAR="${DL}/train-clean-100.tar.gz"
TC100_DST="${RAW}/english/librispeech/LibriSpeech/train-clean-100"
if [[ ! -d "$TC100_DST" ]]; then
  download "https://openslr.org/resources/12/train-clean-100.tar.gz" "$TC100_TAR"
  echo "[extract] train-clean-100 ..."
  tar -xzf "$TC100_TAR" -C "${RAW}/english/librispeech/LibriSpeech" --strip-components=0
  rm -f "$TC100_TAR"
fi

# --- AISHELL-1 ---
AISHELL_TGZ="${DL}/data_aishell.tgz"
AISHELL_ROOT="${RAW}/chinese/aishell"
AISHELL_TRAIN="${AISHELL_ROOT}/data_aishell/wav/train"
if [[ ! -d "$AISHELL_TRAIN" ]] || [[ $(find "$AISHELL_TRAIN" -name '*.wav' 2>/dev/null | head -1 | wc -l) -eq 0 ]]; then
  download "https://openslr.org/resources/33/data_aishell.tgz" "$AISHELL_TGZ"
  echo "[extract] AISHELL outer archive ..."
  mkdir -p "$AISHELL_ROOT"
  tar -xzf "$AISHELL_TGZ" -C "$AISHELL_ROOT"
  rm -f "$AISHELL_TGZ"
  echo "[extract] AISHELL per-speaker archives ..."
  AISHELL_WAV="$AISHELL_ROOT/data_aishell/wav"
  for inner in "$AISHELL_WAV"/S*.tar.gz; do
    [[ -f "$inner" ]] || continue
    tar -xzf "$inner" -C "$AISHELL_WAV"
    rm -f "$inner"
  done
fi

# --- MUSAN ---
MUSAN_TAR="${DL}/musan.tar.gz"
MUSAN_NOISE="${RAW}/noise/musan/noise"
if [[ ! -d "$MUSAN_NOISE" ]]; then
  download "https://openslr.org/resources/17/musan.tar.gz" "$MUSAN_TAR"
  echo "[extract] MUSAN ..."
  mkdir -p "${RAW}/noise"
  tar -xzf "$MUSAN_TAR" -C "${RAW}/noise"
  rm -f "$MUSAN_TAR"
fi

echo "[done] dataset layout:"
ls -la "$TC100_DST" 2>/dev/null | head -3 || true
echo "AISHELL wav count: $(find "$AISHELL_TRAIN" -name '*.wav' 2>/dev/null | wc -l)"
echo "MUSAN noise files: $(find "$MUSAN_NOISE" -name '*.wav' 2>/dev/null | wc -l)"
