from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _parse_snr_conditions(values: list[Any]) -> list[str | float]:
    out: list[str | float] = []
    for value in values:
        if isinstance(value, str) and value.lower() == "clean":
            out.append("clean")
        else:
            out.append(float(value))
    return out


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class ExperimentConfig:
    """实验参数。

    - work_root：仅数据（raw、processed、downloads 等）
    - project_root：仓库内产物（checkpoints、outputs）；绝对路径原样使用
    """

    work_root: Path = Path("data")
    project_root: Path | None = None
    auto_discover_data: bool = True
    clean_roots: dict[str, str] = field(default_factory=dict)
    noise_root: str | None = None

    processed_subdir: str = "processed/mixtures"
    output_subdir: str = "outputs"
    checkpoint_subdir: str = "checkpoints"

    sample_rate: int = 16000
    duration_range_sec: tuple[float, float] = (3.0, 8.0)
    max_offset_sec: float = 2.0
    target_enrollment_sec: float = 5.0
    speaker_counts: list[int] = field(default_factory=lambda: [2, 3, 4])
    noise_snr_db: list[str | float] = field(default_factory=lambda: ["clean", 5.0, 0.0, -5.0])
    min_files_per_speaker: int = 25
    min_speakers_in_pool: int = 3
    num_samples: int = 5000
    random_seed: int = 2026
    balanced_sampling: bool = True

    # MossFormer2 盲分离（测试集基线 + Oracle 上界）
    separation_model: str = "MossFormer2_SS_16K"
    separation_checkpoint_dir: str = "clearvoice"
    separation_output_subdir: str = "separated/mossformer2"
    separation_metrics_name: str = "MossFormer2"

    # ECAPA 说话人编码器（冻结，用于 AttentionTSE 条件与基线选择）
    encoder_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    encoder_savedir: str = "spkrec-ecapa-voxceleb"
    selection_output_subdir: str = "target_extracted/ecapa_mossformer2"
    selection_metrics_name: str = "ECAPA+MossFormer2"

    # AttentionTSE（方案 B 核心模型）
    tse_hidden_dim: int = 256
    tse_lstm_layers: int = 2
    tse_spk_dim: int = 192
    tse_n_fft: int = 512
    tse_hop_length: int = 128
    tse_checkpoint_dir: str = "attention_tse"
    tse_output_subdir: str = "target_extracted/attention_tse"
    tse_metrics_name: str = "AttentionTSE"
    tse_use_wavlm: bool = True
    tse_wavlm_backend: str = "torchaudio"
    tse_wavlm_source: str = "WAVLM_BASE_PLUS"
    tse_wavlm_dim: int = 768
    tse_mag_loss_weight: float = 0.05
    tse_energy_loss_weight: float = 0.0

    spex_checkpoint_dir: str = "spex_plus"
    spex_output_subdir: str = "target_extracted/spex_plus"
    spex_metrics_name: str = "SpEx+"
    spex_spk_loss_weight: float = 0.1
    spex_crop_sec: float = 2.0
    spex_batch_size: int = 8
    spex_train_lr: float = 5e-5
    spex_energy_loss_weight: float = 0.0

    pretrained_tse_model_dir: str = "wesep_bsrnn"
    pretrained_tse_output_subdir: str = "target_extracted/pretrained_tse"
    pretrained_tse_metrics_name: str = "WeSep-BSRNN(预训练)"

    pretrained_tse_ft_checkpoint_dir: str = "wesep_bsrnn_ft"
    pretrained_tse_ft_output_subdir: str = "target_extracted/pretrained_tse_ft"
    pretrained_tse_ft_metrics_name: str = "WeSep-BSRNN(微调)"
    wesep_ft_lr: float = 1e-4
    wesep_ft_epochs: int = 15
    wesep_ft_crop_sec: float = 4.0
    wesep_ft_batch_size: int = 4
    wesep_ft_patience: int = 5

    oracle_metrics_name: str = "Oracle+MossFormer2"

    # 训练
    train_crop_sec: float = 4.0
    train_batch_size: int = 10
    train_lr: float = 5e-4
    train_epochs: int = 100
    train_patience: int = 12
    train_grad_clip: float = 5.0

    device: str = "cuda:0"
    max_samples: int = 0

    @classmethod
    def from_json(cls, path: Path | str) -> ExperimentConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as f:
            raw = json.load(f)
        cfg = cls.from_dict(raw)
        if cfg.project_root is None:
            cfg.project_root = config_path.parent.parent
        if not cfg.work_root.is_absolute():
            cfg.work_root = (cfg.project_root / cfg.work_root).resolve()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        duration = raw.get("duration_range_sec", [3.0, 8.0])
        snr = raw.get("noise_snr_db", ["clean", 5, 0, -5])
        project_root_raw = raw.get("project_root")
        project_root = Path(project_root_raw) if project_root_raw else None
        return cls(
            work_root=Path(raw.get("work_root", "data")),
            project_root=project_root,
            auto_discover_data=bool(raw.get("auto_discover_data", True)),
            clean_roots=dict(raw.get("clean_roots", {})),
            noise_root=raw.get("noise_root"),
            processed_subdir=raw.get("processed_subdir", "processed/mixtures"),
            output_subdir=raw.get("output_subdir", "outputs"),
            checkpoint_subdir=raw.get("checkpoint_subdir", "checkpoints"),
            sample_rate=int(raw.get("sample_rate", 16000)),
            duration_range_sec=(float(duration[0]), float(duration[1])),
            max_offset_sec=float(raw.get("max_offset_sec", 2.0)),
            target_enrollment_sec=float(raw.get("target_enrollment_sec", 5.0)),
            speaker_counts=[int(x) for x in raw.get("speaker_counts", [2, 3, 4])],
            noise_snr_db=_parse_snr_conditions(snr),
            min_files_per_speaker=int(
                raw.get("min_files_per_speaker", raw.get("min_required_utterances_per_speaker", 25))
            ),
            min_speakers_in_pool=int(raw.get("min_speakers_in_pool", 3)),
            num_samples=int(raw.get("num_samples", raw.get("default_num_mixtures", 5000))),
            random_seed=int(raw.get("random_seed", 2026)),
            balanced_sampling=bool(raw.get("balanced_sampling", True)),
            separation_model=str(raw.get("separation_model", "MossFormer2_SS_16K")),
            separation_checkpoint_dir=str(raw.get("separation_checkpoint_dir", "clearvoice")),
            separation_output_subdir=str(raw.get("separation_output_subdir", "separated/mossformer2")),
            separation_metrics_name=str(raw.get("separation_metrics_name", "MossFormer2")),
            encoder_source=str(raw.get("encoder_source", "speechbrain/spkrec-ecapa-voxceleb")),
            encoder_savedir=str(raw.get("encoder_savedir", "spkrec-ecapa-voxceleb")),
            selection_output_subdir=str(
                raw.get("selection_output_subdir", "target_extracted/ecapa_mossformer2")
            ),
            selection_metrics_name=str(raw.get("selection_metrics_name", "ECAPA+MossFormer2")),
            tse_hidden_dim=int(raw.get("tse_hidden_dim", 256)),
            tse_lstm_layers=int(raw.get("tse_lstm_layers", 2)),
            tse_spk_dim=int(raw.get("tse_spk_dim", 192)),
            tse_n_fft=int(raw.get("tse_n_fft", 512)),
            tse_hop_length=int(raw.get("tse_hop_length", 128)),
            tse_checkpoint_dir=str(raw.get("tse_checkpoint_dir", "attention_tse")),
            tse_output_subdir=str(raw.get("tse_output_subdir", "target_extracted/attention_tse")),
            tse_metrics_name=str(raw.get("tse_metrics_name", "AttentionTSE")),
            tse_use_wavlm=bool(raw.get("tse_use_wavlm", True)),
            tse_wavlm_backend=str(raw.get("tse_wavlm_backend", "torchaudio")),
            tse_wavlm_source=str(raw.get("tse_wavlm_source", "WAVLM_BASE_PLUS")),
            tse_wavlm_dim=int(raw.get("tse_wavlm_dim", 768)),
            tse_mag_loss_weight=float(raw.get("tse_mag_loss_weight", 0.05)),
            tse_energy_loss_weight=float(raw.get("tse_energy_loss_weight", 0.0)),
            spex_checkpoint_dir=str(raw.get("spex_checkpoint_dir", "spex_plus")),
            spex_output_subdir=str(raw.get("spex_output_subdir", "target_extracted/spex_plus")),
            spex_metrics_name=str(raw.get("spex_metrics_name", "SpEx+")),
            spex_spk_loss_weight=float(raw.get("spex_spk_loss_weight", 0.1)),
            spex_crop_sec=float(raw.get("spex_crop_sec", 2.0)),
            spex_batch_size=int(raw.get("spex_batch_size", 8)),
            spex_train_lr=float(raw.get("spex_train_lr", 5e-5)),
            spex_energy_loss_weight=float(raw.get("spex_energy_loss_weight", 0.0)),
            pretrained_tse_model_dir=str(
                raw.get(
                    "pretrained_tse_model_dir",
                    raw.get("pretrained_spex_log_subdir", "wesep_bsrnn"),
                )
            ),
            pretrained_tse_output_subdir=str(
                raw.get(
                    "pretrained_tse_output_subdir",
                    raw.get("pretrained_spex_output_subdir", "target_extracted/pretrained_tse"),
                )
            ),
            pretrained_tse_metrics_name=str(
                raw.get(
                    "pretrained_tse_metrics_name",
                    raw.get("pretrained_spex_metrics_name", "WeSep-BSRNN(预训练)"),
                )
            ),
            pretrained_tse_ft_checkpoint_dir=str(
                raw.get("pretrained_tse_ft_checkpoint_dir", "wesep_bsrnn_ft")
            ),
            pretrained_tse_ft_output_subdir=str(
                raw.get("pretrained_tse_ft_output_subdir", "target_extracted/pretrained_tse_ft")
            ),
            pretrained_tse_ft_metrics_name=str(
                raw.get("pretrained_tse_ft_metrics_name", "WeSep-BSRNN(微调)")
            ),
            wesep_ft_lr=float(raw.get("wesep_ft_lr", 1e-4)),
            wesep_ft_epochs=int(raw.get("wesep_ft_epochs", 15)),
            wesep_ft_crop_sec=float(raw.get("wesep_ft_crop_sec", 4.0)),
            wesep_ft_batch_size=int(raw.get("wesep_ft_batch_size", 4)),
            wesep_ft_patience=int(raw.get("wesep_ft_patience", 5)),
            oracle_metrics_name=str(raw.get("oracle_metrics_name", "Oracle+MossFormer2")),
            train_crop_sec=float(raw.get("train_crop_sec", 4.0)),
            train_batch_size=int(raw.get("train_batch_size", 10)),
            train_lr=float(raw.get("train_lr", 5e-4)),
            train_epochs=int(raw.get("train_epochs", 100)),
            train_patience=int(raw.get("train_patience", 12)),
            train_grad_clip=float(raw.get("train_grad_clip", 5.0)),
            device=str(raw.get("device", "cuda:0")),
            max_samples=int(raw.get("max_samples", 0)),
        )

    def _resolved_project_root(self) -> Path:
        return self.project_root if self.project_root is not None else _default_project_root()

    def _resolve_work(self, subpath: str | Path) -> Path:
        path = Path(subpath)
        return path if path.is_absolute() else self.work_root / path

    def _resolve_project(self, subpath: str | Path) -> Path:
        path = Path(subpath)
        return path if path.is_absolute() else self._resolved_project_root() / path

    @property
    def processed_root(self) -> Path:
        return self._resolve_work(self.processed_subdir)

    @property
    def output_root(self) -> Path:
        return self._resolve_project(self.output_subdir)

    @property
    def checkpoint_root(self) -> Path:
        return self._resolve_project(self.checkpoint_subdir)

    @property
    def metadata_path(self) -> Path:
        return self.processed_root / "metadata.csv"

    @property
    def separation_output_root(self) -> Path:
        return self.output_root / self.separation_output_subdir

    @property
    def selection_output_root(self) -> Path:
        return self.output_root / self.selection_output_subdir

    @property
    def tse_output_root(self) -> Path:
        return self.output_root / self.tse_output_subdir

    @property
    def separation_checkpoint_path(self) -> Path:
        p = Path(self.separation_checkpoint_dir)
        return p if p.is_absolute() else self.checkpoint_root / p

    @property
    def encoder_checkpoint_dir(self) -> Path:
        p = Path(self.encoder_savedir)
        return p if p.is_absolute() else self.checkpoint_root / p

    @property
    def tse_checkpoint_path(self) -> Path:
        p = Path(self.tse_checkpoint_dir)
        base = p if p.is_absolute() else self.checkpoint_root / p
        return base / "best.pt"

    @property
    def spex_output_root(self) -> Path:
        return self.output_root / self.spex_output_subdir

    @property
    def spex_checkpoint_path(self) -> Path:
        p = Path(self.spex_checkpoint_dir)
        base = p if p.is_absolute() else self.checkpoint_root / p
        return base / "best.pt"

    @property
    def pretrained_tse_model_path(self) -> Path:
        p = Path(self.pretrained_tse_model_dir)
        return p if p.is_absolute() else self.checkpoint_root / p

    @property
    def pretrained_tse_checkpoint_path(self) -> Path:
        return self.pretrained_tse_model_path / "avg_model.pt"

    @property
    def pretrained_tse_config_path(self) -> Path:
        return self.pretrained_tse_model_path / "config.yaml"

    @property
    def pretrained_tse_output_root(self) -> Path:
        return self.output_root / self.pretrained_tse_output_subdir

    @property
    def pretrained_tse_ft_checkpoint_path(self) -> Path:
        p = Path(self.pretrained_tse_ft_checkpoint_dir)
        base = p if p.is_absolute() else self.checkpoint_root / p
        return base / "best.pt"

    @property
    def pretrained_tse_ft_output_root(self) -> Path:
        return self.output_root / self.pretrained_tse_ft_output_subdir

    def resolved_noise_root(self) -> Path | None:
        if self.noise_root:
            path = self._resolve_work(self.noise_root)
            return path if path.is_dir() else None
        if not self.auto_discover_data:
            return None
        raw = self.work_root / "raw"
        for candidate in (
            raw / "noise" / "musan" / "noise",
            raw / "noise" / "thchs_resource",
        ):
            if candidate.is_dir():
                return candidate
        return None

    def resolved_clean_roots(self) -> dict[str, Path]:
        if self.clean_roots:
            return {name: self._resolve_work(path) for name, path in self.clean_roots.items()}

        if not self.auto_discover_data:
            raise ValueError("未配置 clean_roots，且 auto_discover_data=false")

        raw = self.work_root / "raw"
        candidates = {
            "english": raw / "english" / "librispeech" / "LibriSpeech" / "test-clean",
            "chinese": raw / "chinese" / "thchs30" / "data_thchs30" / "train",
            "diy": raw / "diy" / "recordings",
        }
        roots: dict[str, Path] = {}
        for name, path in candidates.items():
            if not path.is_dir():
                continue
            if name == "diy":
                from cocktail_party.audio import audio_files

                if not any(audio_files(path)):
                    continue
            roots[name] = path
        if not roots:
            raise FileNotFoundError(
                f"在 {raw} 下未找到可用干净语音目录，请配置 clean_roots 或先准备数据"
            )
        return roots
