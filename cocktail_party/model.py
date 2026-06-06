"""方案 B：说话人条件交叉注意力掩码网络（AttentionTSE）。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocktail_party.config import ExperimentConfig


def checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """保存 checkpoint 时排除冻结的 WavLM 权重。"""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("wavlm.")}


def _load_wavlm_torchaudio(bundle_name: str) -> nn.Module:
    import torchaudio

    bundle = getattr(torchaudio.pipelines, bundle_name)
    wavlm = bundle.get_model()
    wavlm.eval()
    for param in wavlm.parameters():
        param.requires_grad = False
    return wavlm


def _load_wavlm_transformers(model_id: str) -> nn.Module:
    from transformers import WavLMModel

    wavlm = WavLMModel.from_pretrained(model_id)
    wavlm.eval()
    for param in wavlm.parameters():
        param.requires_grad = False
    return wavlm


def _wavlm_num_layers(wavlm: nn.Module, backend: str, wavlm_dim: int) -> int:
    if backend == "torchaudio":
        with torch.no_grad():
            dummy = torch.zeros(1, 1600)
            outputs, _ = wavlm.extract_features(dummy)
        return len(outputs)
    with torch.no_grad():
        cfg_layers = getattr(getattr(wavlm, "config", None), "num_hidden_layers", 12)
    return int(cfg_layers) + 1


class AttentionTSE(nn.Module):
    """混合语谱图 + 说话人嵌入 → 软掩码 → 目标说话人波形。"""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        n_freq: int = 257,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        spk_dim: int = 192,
        *,
        use_wavlm: bool = True,
        wavlm_backend: str = "torchaudio",
        wavlm_source: str = "WAVLM_BASE_PLUS",
        wavlm_dim: int = 768,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freq = n_freq
        self.hidden_dim = hidden_dim
        self.lstm_out_dim = hidden_dim * 2
        self.use_wavlm = use_wavlm
        self.wavlm_backend = wavlm_backend

        self.freq_encoder = nn.Sequential(
            nn.Conv1d(n_freq, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.film_gamma = nn.Linear(spk_dim, hidden_dim)
        self.film_beta = nn.Linear(spk_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.query_proj = nn.Linear(spk_dim, self.lstm_out_dim)
        self.key_proj = nn.Linear(self.lstm_out_dim, self.lstm_out_dim)
        self.value_proj = nn.Linear(self.lstm_out_dim, self.lstm_out_dim)
        self.mask_head = nn.Sequential(
            nn.Linear(self.lstm_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_freq),
        )
        nn.init.constant_(self.mask_head[-1].bias, 1.0)

        self.wavlm: nn.Module | None = None
        self.wavlm_layer_weights: nn.Parameter | None = None
        self.wavlm_proj: nn.Linear | None = None
        if use_wavlm:
            if wavlm_backend == "transformers":
                self.wavlm = _load_wavlm_transformers(wavlm_source)
            else:
                self.wavlm = _load_wavlm_torchaudio(wavlm_source)
            n_layers = _wavlm_num_layers(self.wavlm, wavlm_backend, wavlm_dim)
            self.wavlm_layer_weights = nn.Parameter(torch.zeros(n_layers))
            self.wavlm_proj = nn.Linear(wavlm_dim, hidden_dim)

        self.register_buffer("window", torch.hann_window(win_length))

    def train(self, mode: bool = True) -> "AttentionTSE":
        super().train(mode)
        if self.wavlm is not None:
            self.wavlm.eval()
        return self

    def stft(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """waveform: (B, L) -> magnitude (B, T, F), phase (B, T, F)"""
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            return_complex=True,
            center=True,
        )
        magnitude = spec.abs().transpose(1, 2).contiguous()
        phase = spec.angle().transpose(1, 2).contiguous()
        return magnitude, phase

    def istft(self, magnitude: torch.Tensor, phase: torch.Tensor, length: int) -> torch.Tensor:
        """magnitude/phase: (B, T, F) -> waveform (B, L)"""
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        spec = torch.complex(real, imag).transpose(1, 2)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(magnitude.device),
            length=length,
            center=True,
        )

    def _wavlm_layer_outputs(self, mixture: torch.Tensor) -> torch.Tensor:
        """返回 WavLM 各层加权融合特征 (B, T_w, wavlm_dim)。"""
        assert self.wavlm is not None and self.wavlm_layer_weights is not None

        mean = mixture.mean(dim=-1, keepdim=True)
        std = mixture.std(dim=-1, keepdim=True) + 1e-7
        w_in = (mixture - mean) / std

        with torch.no_grad():
            if self.wavlm_backend == "transformers":
                outputs = self.wavlm(w_in, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                stack = torch.stack(hidden_states, dim=0)
            else:
                layer_list, _ = self.wavlm.extract_features(w_in)
                stack = torch.stack(layer_list, dim=0)

        weights = torch.softmax(self.wavlm_layer_weights, dim=0).view(-1, 1, 1, 1)
        return (stack * weights).sum(dim=0)

    def _wavlm_features(self, mixture: torch.Tensor, time_steps: int) -> torch.Tensor:
        assert self.wavlm_proj is not None
        wfeat = self._wavlm_layer_outputs(mixture)
        wfeat = F.interpolate(
            wfeat.transpose(1, 2),
            size=time_steps,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return self.wavlm_proj(wfeat)

    def forward(self, mixture: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mixture: (B, L) 混合波形
            speaker_emb: (B, spk_dim) ECAPA 说话人嵌入
        Returns:
            (B, L) 估计的目标说话人波形
        """
        batch, length = mixture.shape
        speaker_emb = F.normalize(speaker_emb, p=2, dim=-1)
        mix_mag, mix_phase = self.stft(mixture)

        x = mix_mag.transpose(1, 2)
        x = self.freq_encoder(x)
        x = x.transpose(1, 2)

        if self.use_wavlm:
            x = x + self._wavlm_features(mixture, x.size(1))

        gamma = self.film_gamma(speaker_emb).unsqueeze(1)
        beta = self.film_beta(speaker_emb).unsqueeze(1)
        x = gamma * x + beta

        lstm_out, _ = self.lstm(x)

        q = self.query_proj(speaker_emb).unsqueeze(1)
        k = self.key_proj(lstm_out)
        v = self.value_proj(lstm_out)
        scale = math.sqrt(self.lstm_out_dim)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / scale, dim=-1)
        context = torch.matmul(attn, v)
        fused = lstm_out + context.expand_as(lstm_out)

        mask = torch.sigmoid(self.mask_head(fused))
        est_mag = mask * mix_mag
        return self.istft(est_mag, mix_phase, length)


def si_sdr_loss(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """可微 SI-SDR 损失（返回负值，最小化即最大化 SI-SDR）。"""
    est = estimate - estimate.mean(dim=-1, keepdim=True)
    ref = reference - reference.mean(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps
    projection = (est * ref).sum(dim=-1, keepdim=True) * ref / ref_energy
    noise = est - projection
    ratio = (projection.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps)
    ratio = ratio.clamp(max=1e4)
    si_sdr = 10.0 * torch.log10(ratio + eps).squeeze(-1)
    silent = est.pow(2).mean(dim=-1) < 1e-10
    si_sdr = torch.where(silent, torch.full_like(si_sdr, -50.0), si_sdr)
    return -si_sdr.mean()


def mag_l1_loss(
    model: AttentionTSE,
    estimate: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """log-magnitude L1，强化频域抑制干扰。"""
    est_mag, _ = model.stft(estimate)
    with torch.no_grad():
        tgt_mag, _ = model.stft(reference)
    return F.l1_loss(torch.log1p(est_mag), torch.log1p(tgt_mag))


def build_attention_tse(cfg: ExperimentConfig) -> AttentionTSE:
    n_freq = cfg.tse_n_fft // 2 + 1
    return AttentionTSE(
        n_fft=cfg.tse_n_fft,
        hop_length=cfg.tse_hop_length,
        win_length=cfg.tse_n_fft,
        n_freq=n_freq,
        hidden_dim=cfg.tse_hidden_dim,
        lstm_layers=cfg.tse_lstm_layers,
        spk_dim=cfg.tse_spk_dim,
        use_wavlm=cfg.tse_use_wavlm,
        wavlm_backend=cfg.tse_wavlm_backend,
        wavlm_source=cfg.tse_wavlm_source,
        wavlm_dim=cfg.tse_wavlm_dim,
    )


def load_attention_tse(cfg: ExperimentConfig, device: str | torch.device) -> AttentionTSE:
    model = build_attention_tse(cfg)
    ckpt_path = cfg.tse_checkpoint_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"未找到 AttentionTSE 权重：{ckpt_path}。请先运行 train 阶段。"
        )
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
