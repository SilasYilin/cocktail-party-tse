"""SpEx+ 风格时域端到端目标说话人提取（多尺度编码 + 说话人融合）。"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocktail_party.config import ExperimentConfig
from cocktail_party.model import si_sdr_loss


def _mag_l1_loss(est: torch.Tensor, ref: torch.Tensor, n_fft: int = 512, hop: int = 128) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=est.device)
    est_mag = torch.stft(est, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs()
    with torch.no_grad():
        ref_mag = torch.stft(ref, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs()
    return F.l1_loss(torch.log1p(est_mag), torch.log1p(ref_mag))


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpExPlus(nn.Module):
    """混合波形 + ECAPA 嵌入 → 目标说话人波形（多尺度时域掩码/重建）。"""

    def __init__(
        self,
        hidden_dim: int = 256,
        spk_dim: int = 192,
        n_scales: int = 3,
        num_speakers: int = 512,
    ) -> None:
        super().__init__()
        kernels = [15, 31, 63][:n_scales]
        self.branches = nn.ModuleList(
            [_ConvBlock(1, hidden_dim // n_scales, k) for k in kernels]
        )
        fuse_in = (hidden_dim // n_scales) * len(kernels)
        self.fuse = nn.Conv1d(fuse_in, hidden_dim, kernel_size=1)
        self.spk_gamma = nn.Linear(spk_dim, hidden_dim)
        self.spk_beta = nn.Linear(spk_dim, hidden_dim)
        self.tcn = nn.Sequential(
            _ConvBlock(hidden_dim, hidden_dim, 15),
            _ConvBlock(hidden_dim, hidden_dim, 15),
            _ConvBlock(hidden_dim, hidden_dim, 15),
        )
        self.mask_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, 1),
        )
        nn.init.constant_(self.mask_head[-1].bias, 1.0)
        self.spk_classifier = nn.Linear(hidden_dim, num_speakers)

    def encode_mixture(self, mixture: torch.Tensor) -> torch.Tensor:
        x = mixture.unsqueeze(1)
        feats = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(feats, dim=1))

    def forward(self, mixture: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        speaker_emb = F.normalize(speaker_emb, p=2, dim=-1)
        h = self.encode_mixture(mixture)
        gamma = self.spk_gamma(speaker_emb).unsqueeze(-1)
        beta = self.spk_beta(speaker_emb).unsqueeze(-1)
        h = gamma * h + beta
        h = self.tcn(h)
        mask = torch.sigmoid(self.mask_head(h)).squeeze(1)
        return mixture * mask

    def speaker_logits(self, mixture: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        h = self.encode_mixture(mixture)
        pooled = h.mean(dim=-1)
        return self.spk_classifier(pooled)


def build_spex(cfg: ExperimentConfig, num_speakers: int = 512) -> SpExPlus:
    return SpExPlus(
        hidden_dim=cfg.tse_hidden_dim,
        spk_dim=cfg.tse_spk_dim,
        num_speakers=num_speakers,
    )


def load_spex(
    cfg: ExperimentConfig,
    device: str | torch.device,
    num_speakers: int = 512,
) -> SpExPlus:
    model = build_spex(cfg, num_speakers=num_speakers)
    ckpt = cfg.spex_checkpoint_path
    if not ckpt.is_file():
        raise FileNotFoundError(f"未找到 SpEx+ 权重：{ckpt}。请先运行 train_spex 阶段。")
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def spex_training_loss(
    model: SpExPlus,
    mixture: torch.Tensor,
    target: torch.Tensor,
    spk: torch.Tensor,
    cfg: ExperimentConfig,
    speaker_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    est = model(mixture, spk)
    loss = si_sdr_loss(est, target)
    energy_w = cfg.spex_energy_loss_weight
    if energy_w > 0:
        est_rms = est.pow(2).mean(dim=-1).sqrt().clamp(min=1e-4, max=1.0)
        loss = loss + energy_w * (-torch.log(est_rms).mean())
    if cfg.spex_spk_loss_weight > 0 and speaker_ids is not None:
        logits = model.speaker_logits(mixture, spk)
        loss = loss + cfg.spex_spk_loss_weight * F.cross_entropy(logits, speaker_ids)
    return loss
