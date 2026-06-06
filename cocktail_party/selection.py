"""ECAPA 说话人嵌入：用于 AttentionTSE 条件与 MossFormer2 基线选择。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cocktail_party.audio import read_audio_mono, write_wav
from cocktail_party.config import ExperimentConfig
from cocktail_party.io import find_separation_estimates, read_metadata

_ENCODER_FILES = (
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
)


def _encoder_weights_complete(savedir: Path) -> bool:
    return all((savedir / name).is_file() for name in _ENCODER_FILES)


def _ensure_local_pretrained_path(savedir: Path) -> None:
    hyperparams = savedir / "hyperparams.yaml"
    if not hyperparams.is_file() or not _encoder_weights_complete(savedir):
        return
    text = hyperparams.read_text(encoding="utf-8")
    local = savedir.resolve().as_posix()
    marker = f"pretrained_path: {local}"
    if marker in text:
        return
    lines: list[str] = []
    replaced = False
    for line in text.splitlines():
        if not replaced and line.startswith("pretrained_path:"):
            lines.append(marker)
            replaced = True
        else:
            lines.append(line)
    if replaced:
        hyperparams.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_encoder(cfg: ExperimentConfig, device: str):
    savedir = cfg.encoder_checkpoint_dir
    hyperparams = savedir / "hyperparams.yaml"
    if hyperparams.is_file():
        _ensure_local_pretrained_path(savedir)
        source = str(savedir)
    else:
        hub = cfg.encoder_source
        source = hub if "/" in hub else f"speechbrain/{hub}"
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:
        from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source=source,
        savedir=str(savedir),
        run_opts={"device": device},
    )


def embed_audio(encoder, audio: np.ndarray, device: str) -> torch.Tensor:
    waveform = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(waveform)
    return emb.squeeze().detach().cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def run_selection_ecapa(
    cfg: ExperimentConfig,
    separation_root: Path,
    output_root: Path,
    model_name: str = "ECAPA+MossFormer2",
    rows: list[dict[str, str]] | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt = cfg.encoder_checkpoint_dir
    if not (ckpt / "hyperparams.yaml").is_file():
        raise FileNotFoundError(
            f"未找到说话人编码器权重：{ckpt / 'hyperparams.yaml'}。"
            f"请将 ECAPA 权重放在 {ckpt}。"
        )
    encoder = load_encoder(cfg, cfg.device)
    if rows is None:
        rows = read_metadata(cfg.metadata_path)
    if cfg.max_samples > 0:
        rows = rows[: cfg.max_samples]

    decisions: list[dict[str, str]] = []
    for row in tqdm(rows, desc=f"selection ({model_name})"):
        estimate_paths = find_separation_estimates(separation_root, row["sample_id"])
        if not estimate_paths:
            continue
        enrollment, _ = read_audio_mono(Path(row["enrollment_path"]), cfg.sample_rate)
        enrollment_emb = embed_audio(encoder, enrollment, cfg.device)
        scored: list[tuple[float, Path]] = []
        for path in estimate_paths:
            audio, _ = read_audio_mono(path, cfg.sample_rate)
            emb = embed_audio(encoder, audio, cfg.device)
            scored.append((cosine(enrollment_emb, emb), path))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path = scored[0]
        output_path = output_root / f"{row['sample_id']}.wav"
        audio, _ = read_audio_mono(best_path, cfg.sample_rate)
        write_wav(output_path, audio, cfg.sample_rate)
        decisions.append(
            {
                "sample_id": row["sample_id"],
                "target_speaker": row["target_speaker"],
                "selected_separation": str(best_path),
                "output_path": str(output_path),
                "score": f"{best_score:.6f}",
                "n_candidates": str(len(scored)),
            }
        )

    decisions_path = output_root / "selection_decisions.csv"
    if decisions:
        with decisions_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(decisions[0].keys()))
            writer.writeheader()
            writer.writerows(decisions)
    return output_root
