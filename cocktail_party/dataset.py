"""训练/验证数据集与说话人划分。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cocktail_party.audio import crop_or_pad, loop_or_pad, read_audio_mono
from cocktail_party.config import ExperimentConfig
from cocktail_party.io import read_metadata


def speaker_splits_path(cfg: ExperimentConfig) -> Path:
    return cfg.output_root / "speaker_splits.json"


def get_speaker_splits(
    cfg: ExperimentConfig,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict:
    """按目标说话人做 speaker-disjoint 划分，并写入 speaker_splits.json。"""
    rows = read_metadata(cfg.metadata_path)
    speakers = sorted({r["target_speaker"] for r in rows})
    rng = np.random.default_rng(cfg.random_seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_spks = speakers[:n_train]
    val_spks = speakers[n_train : n_train + n_val]
    test_spks = speakers[n_train + n_val :]

    train_set = set(train_spks)
    val_set = set(val_spks)
    test_set = set(test_spks)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for row in rows:
        sid = row["sample_id"]
        spk = row["target_speaker"]
        if spk in train_set:
            train_ids.append(sid)
        elif spk in val_set:
            val_ids.append(sid)
        else:
            test_ids.append(sid)

    splits = {
        "train_speakers": train_spks,
        "val_speakers": val_spks,
        "test_speakers": test_spks,
        "train_sample_ids": train_ids,
        "val_sample_ids": val_ids,
        "test_sample_ids": test_ids,
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "speakers_train": len(train_spks),
            "speakers_val": len(val_spks),
            "speakers_test": len(test_spks),
        },
    }

    out_path = speaker_splits_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    return splits


def load_speaker_splits(cfg: ExperimentConfig) -> dict:
    path = speaker_splits_path(cfg)
    if not path.is_file():
        return get_speaker_splits(cfg)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def rows_for_split(cfg: ExperimentConfig, split: str) -> list[dict[str, str]]:
    """split: train | val | test"""
    all_rows = read_metadata(cfg.metadata_path)
    splits = load_speaker_splits(cfg)
    key = f"{split}_sample_ids"
    allowed = set(splits[key])
    return [r for r in all_rows if r["sample_id"] in allowed]


def _crop_aligned_pair(
    mixture: np.ndarray,
    target: np.ndarray,
    crop_samples: int,
    rng: np.random.Generator,
    *,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """mixture 与 target 使用相同时间起点裁剪，避免标签错位。"""
    length = min(mixture.size, target.size)
    mixture = mixture[:length]
    target = target[:length]
    if length > crop_samples:
        start = (
            int(rng.integers(0, length - crop_samples + 1))
            if augment
            else 0
        )
        mixture = mixture[start : start + crop_samples]
        target = target[start : start + crop_samples]
    elif length >= crop_samples:
        mixture = mixture[:crop_samples]
        target = target[:crop_samples]
    else:
        mixture = crop_or_pad(mixture, crop_samples, rng)
        target = crop_or_pad(target, crop_samples, rng)
    return mixture.astype(np.float32), target.astype(np.float32)


def _normalize_pair(
    mixture: np.ndarray, target: np.ndarray, peak: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """按混合音峰值归一化，目标轨同比例缩放。"""
    max_abs = float(max(np.max(np.abs(mixture)), np.max(np.abs(target)), 1e-8))
    scale = peak / max_abs
    return (mixture * scale).astype(np.float32), (target * scale).astype(np.float32)


class CocktailDataset(Dataset):
    """(mixture, enrollment, target) 三元组，固定长度随机裁剪。"""

    def __init__(
        self,
        rows: list[dict[str, str]],
        cfg: ExperimentConfig,
        crop_sec: float | None = None,
        augment: bool = True,
    ) -> None:
        self.rows = rows
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.crop_samples = int(round((crop_sec or cfg.train_crop_sec) * self.sr))
        self.augment = augment
        self._rng = np.random.default_rng(cfg.random_seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        mixture, _ = read_audio_mono(Path(row["mixture_path"]), self.sr)
        target, _ = read_audio_mono(Path(row["target_source_path"]), self.sr)
        mixture, target = _crop_aligned_pair(
            mixture, target, self.crop_samples, self._rng, augment=self.augment
        )
        mixture, target = _normalize_pair(mixture, target)

        enrollment, _ = read_audio_mono(Path(row["enrollment_path"]), self.sr)
        enroll_samples = int(round(self.cfg.target_enrollment_sec * self.sr))
        if enrollment.size > enroll_samples:
            if self.augment:
                start = int(self._rng.integers(0, enrollment.size - enroll_samples + 1))
                enrollment = enrollment[start : start + enroll_samples]
            else:
                enrollment = enrollment[:enroll_samples]
        elif enrollment.size < enroll_samples:
            enrollment = loop_or_pad(enrollment, enroll_samples)

        return {
            "mixture": torch.from_numpy(mixture),
            "enrollment": torch.from_numpy(enrollment.astype(np.float32)),
            "target": torch.from_numpy(target),
            "sample_id": row["sample_id"],
        }
