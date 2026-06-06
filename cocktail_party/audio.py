from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
EPS = 1e-8


def audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


def read_audio_mono(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if sample_rate is not None and sr != sample_rate:
        mono = resample_audio(mono, sr, sample_rate)
        sr = sample_rate
    return mono.astype(np.float32), sr


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate)


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio.astype(np.float32)
    gcd = math.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + EPS))


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs < EPS:
        return audio.astype(np.float32)
    return (audio * (peak / max_abs)).astype(np.float32)


def normalize_rms(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    return (audio * (target_rms / max(rms(audio), EPS))).astype(np.float32)


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def crop_or_pad(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if audio.size == length:
        return audio.astype(np.float32)
    if audio.size > length:
        start = int(rng.integers(0, audio.size - length + 1))
        return audio[start : start + length].astype(np.float32)
    out = np.zeros(length, dtype=np.float32)
    out[: audio.size] = audio
    return out


def loop_or_pad(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.size >= length:
        return audio[:length].astype(np.float32)
    if audio.size == 0:
        return np.zeros(length, dtype=np.float32)
    repeats = int(np.ceil(length / audio.size))
    return np.tile(audio, repeats)[:length].astype(np.float32)


def scale_noise_to_snr(noise: np.ndarray, clean_mix: np.ndarray, snr_db: float) -> np.ndarray:
    clean_power = np.mean(np.square(clean_mix), dtype=np.float64) + EPS
    noise_power = np.mean(np.square(noise), dtype=np.float64) + EPS
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    gain = math.sqrt(target_noise_power / noise_power)
    return (noise * gain).astype(np.float32)


def align_length(items: Iterable[np.ndarray]) -> list[np.ndarray]:
    arrays = [np.asarray(x, dtype=np.float32).reshape(-1) for x in items]
    if not arrays:
        return []
    length = min(x.size for x in arrays)
    return [x[:length] for x in arrays]


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate, reference = align_length([estimate, reference])
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    ref_energy = np.sum(reference**2) + EPS
    projection = np.sum(estimate * reference) * reference / ref_energy
    noise = estimate - projection
    return float(10.0 * np.log10((np.sum(projection**2) + EPS) / (np.sum(noise**2) + EPS)))


def discover_speakers(root: Path) -> dict[str, list[Path]]:
    files = audio_files(root)
    speakers: dict[str, list[Path]] = {}
    for path in files:
        rel = path.relative_to(root)
        if len(rel.parts) >= 2:
            speaker_id = rel.parts[0]
        else:
            speaker_id = path.stem.split("_")[0].split("-")[0]
        speakers.setdefault(speaker_id, []).append(path)
    return {k: sorted(v) for k, v in sorted(speakers.items())}
