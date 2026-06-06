#!/usr/bin/env python3
"""将 DIY 原始录音转为 16 kHz mono WAV 并按句切分。"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from cocktail_party.audio import write_wav

INPUT_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".aiff", ".aif"}
DEFAULT_SR = 16000
SEG_MIN_SEC = 3.0
SEG_MAX_SEC = 8.0
MIN_RMS = 0.005
FRAME_MS = 20
HOP_MS = 10


def slugify(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "utt"


def find_ffmpeg() -> str:
    for candidate in (
        shutil.which("ffmpeg"),
        str(Path(sys.executable).parent / "ffmpeg"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("未找到 ffmpeg。请安装: conda install -c conda-forge ffmpeg")


def decode_to_mono(path: Path, sample_rate: int, ffmpeg: str) -> np.ndarray:
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    if not proc.stdout:
        raise RuntimeError(f"ffmpeg 未输出音频: {path}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def frame_rms(audio: np.ndarray, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    if audio.size < frame:
        return np.array([float(np.sqrt(np.mean(audio**2) + 1e-8))], dtype=np.float32), np.array([0], dtype=np.int64)
    starts = np.arange(0, audio.size - frame + 1, hop, dtype=np.int64)
    rms = np.array(
        [float(np.sqrt(np.mean(audio[s : s + frame] ** 2) + 1e-8)) for s in starts],
        dtype=np.float32,
    )
    return rms, starts


def split_utterances(
    audio: np.ndarray,
    sample_rate: int,
    min_sec: float = SEG_MIN_SEC,
    max_sec: float = SEG_MAX_SEC,
    min_rms: float = MIN_RMS,
) -> list[np.ndarray]:
    frame = int(round(sample_rate * FRAME_MS / 1000.0))
    hop = int(round(sample_rate * HOP_MS / 1000.0))
    min_samples = int(round(min_sec * sample_rate))
    max_samples = int(round(max_sec * sample_rate))

    rms, starts = frame_rms(audio, frame, hop)
    if rms.size == 0:
        return []

    threshold = max(min_rms, float(np.percentile(rms, 25)))
    voiced = rms >= threshold

    segments: list[tuple[int, int]] = []
    i = 0
    while i < len(voiced):
        if not voiced[i]:
            i += 1
            continue
        seg_start = int(starts[i])
        j = i
        while j < len(voiced) and voiced[j]:
            j += 1
        seg_end = int(min(starts[j - 1] + frame if j > i else seg_start + frame, audio.size))
        seg_len = seg_end - seg_start

        if seg_len > max_samples:
            cursor = seg_start
            while cursor < seg_end:
                chunk_end = min(cursor + max_samples, seg_end)
                if chunk_end - cursor >= min_samples:
                    segments.append((cursor, chunk_end))
                elif segments and chunk_end - cursor > sample_rate:
                    prev_start, prev_end = segments[-1]
                    if prev_end - prev_start < max_samples:
                        merged_end = min(prev_end + (chunk_end - cursor), prev_start + max_samples)
                        segments[-1] = (prev_start, merged_end)
                cursor = chunk_end
        elif seg_len >= min_samples:
            segments.append((seg_start, seg_end))
        elif segments:
            prev_start, prev_end = segments[-1]
            if prev_end - prev_start + seg_len <= max_samples and seg_len > sample_rate:
                segments[-1] = (prev_start, seg_end)
        i = j

    if not segments and audio.size >= min_samples:
        step = max_samples
        for start in range(0, audio.size - min_samples + 1, step):
            segments.append((start, min(start + max_samples, audio.size)))

    return [audio[s:e].astype(np.float32) for s, e in segments if e - s >= min_samples]


def import_file(
    path: Path,
    out_dir: Path,
    speaker_id: str,
    sample_rate: int,
    ffmpeg: str,
    min_utterances: int,
    start_index: int,
) -> tuple[int, int]:
    audio = decode_to_mono(path, sample_rate, ffmpeg)
    chunks = split_utterances(audio, sample_rate)
    if not chunks:
        chunks = [audio[: int(round(SEG_MAX_SEC * sample_rate))].astype(np.float32)]

    speaker_dir = out_dir / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    index = start_index
    for chunk in chunks:
        out_path = speaker_dir / f"utt{index:03d}.wav"
        write_wav(out_path, chunk, sample_rate)
        written += 1
        index += 1

    if written < min_utterances and len(chunks) == 1 and chunks[0].size > int(min_utterances * SEG_MIN_SEC * sample_rate * 0.5):
        chunk = chunks[0]
        extra = split_utterances(chunk, sample_rate, min_sec=SEG_MIN_SEC, max_sec=SEG_MAX_SEC, min_rms=0.0)
        if len(extra) > written:
            for i, seg in enumerate(extra, start=start_index):
                out_path = speaker_dir / f"utt{i:03d}.wav"
                write_wav(out_path, seg, sample_rate)
            written = len(extra)
            index = start_index + written

    return written, index


def collect_inputs(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS)


def import_recordings(
    input_dir: Path,
    output_root: Path,
    speaker_id: str = "spk01",
    sample_rate: int = DEFAULT_SR,
    min_utterances: int = 20,
    clear_speaker: bool = False,
) -> Path:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    inputs = collect_inputs(input_dir)
    if not inputs:
        raise FileNotFoundError(f"未在 {input_dir} 找到支持的音频: {sorted(INPUT_EXTENSIONS)}")

    ffmpeg = find_ffmpeg()
    speaker_dir = output_root / speaker_id
    if clear_speaker and speaker_dir.exists():
        for wav in speaker_dir.glob("*.wav"):
            wav.unlink()

    next_index = 1
    if speaker_dir.exists():
        nums = []
        for p in speaker_dir.glob("utt*.wav"):
            m = re.search(r"utt(\d+)", p.stem)
            if m:
                nums.append(int(m.group(1)))
        if nums:
            next_index = max(nums) + 1

    for path in inputs:
        n, next_index = import_file(
            path, output_root, speaker_id, sample_rate, ffmpeg, min_utterances, next_index
        )
        print(f"[ok] {path.name} -> {n} utterance(s)")

    final_count = len(list(speaker_dir.glob("*.wav")))
    print(f"完成: {speaker_id} 共 {final_count} 条 -> {speaker_dir}")
    if final_count < min_utterances:
        print(f"警告: 仅 {final_count} 条，少于 {min_utterances} 条")
    return speaker_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="导入 DIY 录音为测评用 WAV 句段")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw/diy/recordings"),
        help="说话人 WAV 输出目录（相对路径相对于当前工作目录）",
    )
    parser.add_argument("--speaker-id", default="spk01")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    parser.add_argument("--min-utterances", type=int, default=20)
    parser.add_argument("--clear-speaker", action="store_true")
    args = parser.parse_args()
    import_recordings(
        args.input_dir,
        args.output_root,
        args.speaker_id,
        args.sample_rate,
        args.min_utterances,
        args.clear_speaker,
    )


if __name__ == "__main__":
    main()
