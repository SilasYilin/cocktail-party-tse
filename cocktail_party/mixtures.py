from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cocktail_party.audio import (
    audio_files,
    crop_or_pad,
    db_to_gain,
    discover_speakers,
    loop_or_pad,
    normalize_peak,
    normalize_rms,
    read_audio_mono,
    scale_noise_to_snr,
    write_wav,
)
from cocktail_party.config import ExperimentConfig


@dataclass(frozen=True)
class CleanDataset:
    name: str
    root: Path
    speakers: dict[str, list[Path]]


def safe_speaker_label(speaker_key: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", speaker_key.replace("/", "__"))


def speaker_source_type(speaker_key: str) -> str:
    return speaker_key.split("/", 1)[0]


def choose_different_file(files: list[Path], used: Path, rng: np.random.Generator) -> Path:
    candidates = [x for x in files if x != used]
    if not candidates:
        candidates = files
    return candidates[int(rng.integers(0, len(candidates)))]


def load_source_segment(path: Path, sr: int, duration_samples: int, rng: np.random.Generator) -> np.ndarray:
    audio, _ = read_audio_mono(path, sr)
    audio = crop_or_pad(audio, duration_samples, rng)
    return normalize_rms(audio, 0.05)


def load_enrollment(path: Path, sr: int, enrollment_samples: int) -> np.ndarray:
    audio, _ = read_audio_mono(path, sr)
    audio = loop_or_pad(audio, enrollment_samples)
    return normalize_peak(audio, 0.8)


def build_noise(noise_files: list[Path], length: int, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    if not noise_files:
        return np.zeros(length, dtype=np.float32), ""
    path = noise_files[int(rng.integers(0, len(noise_files)))]
    audio, _ = read_audio_mono(path, sr)
    return loop_or_pad(audio, length), str(path)


def load_clean_datasets(clean_roots: dict[str, Path], min_files_per_speaker: int) -> list[CleanDataset]:
    datasets: list[CleanDataset] = []
    for name, root in clean_roots.items():
        speakers = {
            spk: files
            for spk, files in discover_speakers(root).items()
            if len(files) >= min_files_per_speaker
        }
        if speakers:
            datasets.append(CleanDataset(name=name, root=root, speakers=speakers))
    if not datasets:
        raise FileNotFoundError("无可用干净语音数据集（需按说话人分子目录，且每人满足最少句数）")
    return datasets


def build_global_speaker_pool(
    clean_roots: dict[str, Path],
    min_files_per_speaker: int,
) -> dict[str, list[Path]]:
    """合并 english/chinese/diy 的合格说话人为统一池，键为 ``数据源/说话人``。"""
    pool: dict[str, list[Path]] = {}
    for dataset_name, root in clean_roots.items():
        for spk, files in discover_speakers(root).items():
            if len(files) >= min_files_per_speaker:
                pool[f"{dataset_name}/{spk}"] = files
    return pool


def pool_stats(pool: dict[str, list[Path]]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for key in pool:
        src = speaker_source_type(key)
        stats[src] = stats.get(src, 0) + 1
    return stats


def make_one_sample(
    sample_id: str,
    speaker_pool: dict[str, list[Path]],
    selected: list[str],
    n_speakers: int,
    condition: str | float,
    noise_files: list[Path],
    cfg: ExperimentConfig,
    output_root: Path,
    rng: np.random.Generator,
) -> dict[str, str]:
    sr = cfg.sample_rate
    duration_sec = float(rng.uniform(*cfg.duration_range_sec))
    duration_samples = int(round(duration_sec * sr))
    max_offset_samples = int(round(cfg.max_offset_sec * sr))
    target_index = int(rng.integers(0, n_speakers))
    final_length = duration_samples + max_offset_samples

    sources: list[np.ndarray] = []
    source_paths: list[Path] = []
    offsets: list[int] = []
    gains_db: list[float] = []
    for spk in selected:
        files = speaker_pool[spk]
        source_path = files[int(rng.integers(0, len(files)))]
        source = load_source_segment(source_path, sr, duration_samples, rng)
        offset = int(rng.integers(0, max_offset_samples + 1)) if max_offset_samples > 0 else 0
        gain_db = float(rng.uniform(-3.0, 3.0))
        placed = np.zeros(final_length, dtype=np.float32)
        placed[offset : offset + duration_samples] = source * db_to_gain(gain_db)
        sources.append(placed)
        source_paths.append(source_path)
        offsets.append(offset)
        gains_db.append(gain_db)

    clean_mix = np.sum(np.stack(sources, axis=0), axis=0)
    noise_path = ""
    if condition == "clean":
        mixture = clean_mix.copy()
        noise_snr = "clean"
    else:
        noise, noise_path = build_noise(noise_files, clean_mix.size, sr, rng)
        noise = scale_noise_to_snr(noise, clean_mix, float(condition))
        mixture = clean_mix + noise
        noise_snr = str(condition)

    scale = 0.95 / max(float(np.max(np.abs(mixture))), 0.95)
    mixture = (mixture * scale).astype(np.float32)
    sources = [(src * scale).astype(np.float32) for src in sources]

    mix_path = output_root / "mixtures" / f"{sample_id}.wav"
    source_dir = output_root / "clean_sources" / sample_id
    enroll_dir = output_root / "enrollments"
    write_wav(mix_path, mixture, sr)

    source_out_paths: list[str] = []
    for idx, (source, spk) in enumerate(zip(sources, selected), start=1):
        label = safe_speaker_label(spk)
        path = source_dir / f"source{idx}_{label}.wav"
        write_wav(path, source, sr)
        source_out_paths.append(str(path))

    target_spk = selected[target_index]
    target_source_path = source_out_paths[target_index]
    enrollment_file = choose_different_file(speaker_pool[target_spk], source_paths[target_index], rng)
    enrollment = load_enrollment(enrollment_file, sr, int(round(cfg.target_enrollment_sec * sr)))
    enrollment_path = enroll_dir / f"{sample_id}_{safe_speaker_label(target_spk)}_enrollment.wav"
    write_wav(enrollment_path, enrollment, sr)

    source_types = sorted({speaker_source_type(spk) for spk in selected})
    return {
        "sample_id": sample_id,
        "dataset": "mixed",
        "source_types": "|".join(source_types),
        "condition": str(condition),
        "n_speakers": str(n_speakers),
        "sample_rate": str(sr),
        "duration_sec": f"{duration_sec:.3f}",
        "mixture_path": str(mix_path),
        "source_paths": "|".join(source_out_paths),
        "speaker_ids": "|".join(selected),
        "target_speaker": target_spk,
        "target_source_path": target_source_path,
        "enrollment_path": str(enrollment_path),
        "noise_path": noise_path,
        "noise_snr_db": noise_snr,
        "offsets_samples": "|".join(str(x) for x in offsets),
        "source_gain_db": "|".join(f"{x:.2f}" for x in gains_db),
    }


def build_balanced_schedule(
    num_samples: int,
    speaker_counts: list[int],
    conditions: list[str | float],
    noise_files: list[Path],
    rng: np.random.Generator,
) -> list[tuple[int, str | float]]:
    """生成均衡的 (n_speakers, condition) 任务序列，保证每个组合分配均等样本数。"""
    combos: list[tuple[int, str | float]] = []
    for n in speaker_counts:
        for cond in conditions:
            if cond != "clean" and not noise_files:
                combos.append((n, "clean"))
            else:
                combos.append((n, cond))

    n_combos = len(combos)
    base = num_samples // n_combos
    remainder = num_samples % n_combos
    schedule: list[tuple[int, str | float]] = []
    for i, combo in enumerate(combos):
        count = base + (1 if i < remainder else 0)
        schedule.extend([combo] * count)

    rng.shuffle(schedule)
    return schedule


def generate_mixtures(cfg: ExperimentConfig) -> Path:
    rng = np.random.default_rng(cfg.random_seed)
    clean_roots = cfg.resolved_clean_roots()
    speaker_pool = build_global_speaker_pool(clean_roots, cfg.min_files_per_speaker)
    min_pool = cfg.min_speakers_in_pool
    if len(speaker_pool) < min_pool:
        raise RuntimeError(
            f"全局说话人池仅 {len(speaker_pool)} 人，少于要求的 {min_pool} 人"
            f"（每人需 ≥{cfg.min_files_per_speaker} 条干净语音）"
        )

    max_speakers = max(cfg.speaker_counts)
    if len(speaker_pool) < max_speakers:
        raise RuntimeError(
            f"全局说话人池 {len(speaker_pool)} 人，无法满足单条混合最多 {max_speakers} 人"
        )

    conditions = cfg.noise_snr_db
    noise_root = cfg.resolved_noise_root()
    noise_files = audio_files(noise_root) if noise_root else []

    output_root = cfg.processed_root
    output_root.mkdir(parents=True, exist_ok=True)
    pool_keys = list(speaker_pool.keys())

    if cfg.balanced_sampling:
        schedule = build_balanced_schedule(cfg.num_samples, cfg.speaker_counts, conditions, noise_files, rng)
    else:
        schedule = []
        for _ in range(cfg.num_samples):
            n_speakers = int(rng.choice(cfg.speaker_counts))
            condition = conditions[int(rng.integers(0, len(conditions)))]
            if condition != "clean" and not noise_files:
                condition = "clean"
            schedule.append((n_speakers, condition))

    rows: list[dict[str, str]] = []
    for idx, (n_speakers, condition) in enumerate(tqdm(schedule, desc="mixtures"), start=1):
        selected = list(rng.choice(pool_keys, size=n_speakers, replace=False))
        types_tag = "+".join(sorted({speaker_source_type(s) for s in selected}))
        sample_id = f"{idx:04d}_mixed_{types_tag}_{n_speakers}spk_{condition}"
        rows.append(
            make_one_sample(
                sample_id,
                speaker_pool,
                selected,
                n_speakers,
                condition,
                noise_files,
                cfg,
                output_root,
                rng,
            )
        )

    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    datasets = load_clean_datasets(clean_roots, cfg.min_files_per_speaker)
    summary = {
        "num_samples": len(rows),
        "pool_mode": "global",
        "pool_speakers": len(speaker_pool),
        "pool_by_source": pool_stats(speaker_pool),
        "datasets": {ds.name: {"root": str(ds.root), "speakers": len(ds.speakers)} for ds in datasets},
        "conditions": [str(x) for x in conditions],
        "noise_files": len(noise_files),
        "metadata": str(metadata_path),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return metadata_path
