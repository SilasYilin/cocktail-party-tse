from __future__ import annotations

import csv
from pathlib import Path


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_separation_estimates(root: Path, sample_id: str) -> list[Path]:
    sample_dir = root / sample_id
    if sample_dir.exists():
        return sorted(sample_dir.glob("*.wav"))
    return sorted(root.glob(f"{sample_id}*.wav"))


def find_target_estimate(root: Path, sample_id: str) -> Path | None:
    candidates = [
        root / f"{sample_id}.wav",
        root / sample_id / "target.wav",
        root / sample_id / "target_speaker.wav",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.glob(f"{sample_id}*.wav"))
    return matches[0] if matches else None
