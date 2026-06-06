"""WeSep BSRNN+ECAPA（VoxCeleb1，16 kHz）预训练零样本 TSE 推理。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cocktail_party.audio import read_audio_mono, write_wav
from cocktail_party.config import ExperimentConfig
from cocktail_party.dataset import rows_for_split


def load_pretrained_wesep(cfg: ExperimentConfig):
    from wesep.cli.extractor import load_model_local

    model_path = cfg.pretrained_tse_model_path
    if not cfg.pretrained_tse_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"未找到 WeSep 预训练权重：{cfg.pretrained_tse_checkpoint_path}。"
            f"请将 bsrnn_ecapa_vox1 的 avg_model.pt 与 config.yaml 放到 {model_path}/"
        )
    extractor = load_model_local(str(model_path))
    extractor.set_vad(False)
    device = cfg.device if cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    extractor.set_device(device)
    return extractor


def _numpy_to_pcm(wave: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(wave.astype(np.float32)).unsqueeze(0)


def _run_wesep_inference(
    extractor,
    cfg: ExperimentConfig,
    rows: list[dict[str, str]],
    output_root: Path,
    desc: str,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    sr = cfg.sample_rate
    for row in tqdm(rows, desc=desc):
        mixture, _ = read_audio_mono(Path(row["mixture_path"]), sr)
        enrollment, _ = read_audio_mono(Path(row["enrollment_path"]), sr)
        est_t = extractor.extract_speech_from_pcm(
            _numpy_to_pcm(mixture),
            sr,
            _numpy_to_pcm(enrollment),
            sr,
        )
        if est_t is None:
            raise RuntimeError(f"WeSep 未返回有效输出：{row['sample_id']}")
        est = est_t.squeeze(0).numpy()
        if est.ndim > 1:
            est = est[0]
        if est.shape[0] > mixture.shape[0]:
            est = est[: mixture.shape[0]]
        elif est.shape[0] < mixture.shape[0]:
            pad = np.zeros(mixture.shape[0] - est.shape[0], dtype=np.float32)
            est = np.concatenate([est, pad])
        write_wav(output_root / f"{row['sample_id']}.wav", est, sr)
    return output_root


def run_inference_pretrained_tse(
    cfg: ExperimentConfig, rows: list[dict[str, str]] | None = None
) -> Path:
    """混合 + enrollment 波形 → 目标 wav（输出采样率为 cfg.sample_rate）。"""
    if rows is None:
        rows = rows_for_split(cfg, "test")
    if cfg.max_samples > 0:
        rows = rows[: cfg.max_samples]

    extractor = load_pretrained_wesep(cfg)
    return _run_wesep_inference(
        extractor,
        cfg,
        rows,
        cfg.pretrained_tse_output_root,
        "inference (WeSep pretrained)",
    )


def run_inference_pretrained_tse_ft(
    cfg: ExperimentConfig, rows: list[dict[str, str]] | None = None
) -> Path:
    """加载微调权重后在测试集推理。"""
    if rows is None:
        rows = rows_for_split(cfg, "test")
    if cfg.max_samples > 0:
        rows = rows[: cfg.max_samples]
    ft_path = cfg.pretrained_tse_ft_checkpoint_path
    if not ft_path.is_file():
        raise FileNotFoundError(
            f"未找到 WeSep 微调权重：{ft_path}。请先运行 train_wesep 阶段。"
        )
    extractor = load_pretrained_wesep(cfg)
    ckpt = torch.load(ft_path, map_location=extractor.device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    extractor.model.load_state_dict(state, strict=True)
    extractor.model.eval()
    return _run_wesep_inference(
        extractor,
        cfg,
        rows,
        cfg.pretrained_tse_ft_output_root,
        "inference (WeSep finetuned)",
    )
