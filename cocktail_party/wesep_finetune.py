"""WeSep BSRNN 域内微调（冻结 ECAPA，训练分离器）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cocktail_party.config import ExperimentConfig
from cocktail_party.dataset import CocktailDataset, get_speaker_splits, rows_for_split
from cocktail_party.model import si_sdr_loss
from cocktail_party.pretrained_tse import load_pretrained_wesep
from cocktail_party.train import _collate, _device, _trainable_params


def _compute_fbank_batch(
    enrollment: torch.Tensor, sample_rate: int, device: torch.device
) -> torch.Tensor:
    """enrollment: (B, T) -> (B, T_frames, 80)"""
    import torchaudio.compliance.kaldi as kaldi

    feats: list[torch.Tensor] = []
    for i in range(enrollment.size(0)):
        wav = enrollment[i : i + 1].to(device)
        feat = kaldi.fbank(
            wav,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            sample_frequency=sample_rate,
        )
        feat = feat - torch.mean(feat, 0)
        feats.append(feat)
    max_len = max(f.shape[0] for f in feats)
    out = enrollment.new_zeros(len(feats), max_len, 80)
    for i, feat in enumerate(feats):
        out[i, : feat.shape[0]] = feat
    return out


def _set_trainable_separator(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    for name, param in model.named_parameters():
        param.requires_grad = "spk_model" not in name
    return _trainable_params(model)


@torch.no_grad()
def _eval_wesep(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    sample_rate: int,
) -> tuple[float, float]:
    model.eval()
    scores: list[float] = []
    rms_values: list[float] = []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        enrollment = batch["enrollment"].to(device)
        fbank = _compute_fbank_batch(enrollment, sample_rate, device)
        est, _ = model(mixture, fbank)
        scores.append(-float(si_sdr_loss(est, target).item()))
        rms_values.append(float(est.pow(2).mean().sqrt().item()))
    mean_rms = float(np.mean(rms_values)) if rms_values else 0.0
    return float(np.mean(scores)) if scores else float("-inf"), mean_rms


def run_train_wesep(cfg: ExperimentConfig) -> Path:
    get_speaker_splits(cfg)
    device = _device(cfg)
    extractor = load_pretrained_wesep(cfg)
    model = extractor.model
    trainable = _set_trainable_separator(model)

    crop_sec = cfg.wesep_ft_crop_sec
    batch_size = cfg.wesep_ft_batch_size
    train_rows = rows_for_split(cfg, "train")
    val_rows = rows_for_split(cfg, "val")
    train_ds = CocktailDataset(train_rows, cfg, augment=True, crop_sec=crop_sec)
    val_ds = CocktailDataset(val_rows, cfg, augment=False, crop_sec=crop_sec)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    optimizer = torch.optim.Adam(trainable, lr=cfg.wesep_ft_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.wesep_ft_epochs, eta_min=1e-6
    )

    ckpt_dir = cfg.pretrained_tse_ft_checkpoint_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.pretrained_tse_ft_checkpoint_path
    best_val = float("-inf")
    patience_left = cfg.wesep_ft_patience
    sr = cfg.sample_rate

    for epoch in range(1, cfg.wesep_ft_epochs + 1):
        model.train()
        model.spk_model.eval()
        train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"wesep ft epoch {epoch}"):
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            enrollment = batch["enrollment"].to(device)
            fbank = _compute_fbank_batch(enrollment, sr, device)

            optimizer.zero_grad(set_to_none=True)
            est, _ = model(mixture, fbank)
            loss = si_sdr_loss(est, target)
            if not torch.isfinite(loss):
                print("  skip batch: non-finite WeSep FT loss")
                continue
            loss.backward()
            if cfg.train_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.train_grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_si, val_rms = _eval_wesep(model, val_loader, device, sr)
        scheduler.step()
        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        print(
            f"wesep ft epoch {epoch}: train_loss={mean_train:.3f} "
            f"val_si_sdr={val_si:.3f} val_rms={val_rms:.4f}"
        )
        collapsed = val_rms < 1e-5
        if val_si > best_val and not collapsed:
            best_val = val_si
            patience_left = cfg.wesep_ft_patience
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_si_sdr": val_si,
                    "val_rms": val_rms,
                },
                best_path,
            )
            print(f"  saved WeSep FT -> {best_path}")
        elif val_si > best_val and collapsed:
            print(f"  skip WeSep FT checkpoint: collapse (val_rms={val_rms:.2e})")
            patience_left -= 1
        else:
            patience_left -= 1
        if patience_left <= 0:
            print("WeSep FT early stopping")
            break

    if not best_path.is_file():
        torch.save({"model": model.state_dict()}, best_path)
    return best_path
