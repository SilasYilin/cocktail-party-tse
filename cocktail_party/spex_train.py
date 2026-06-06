"""SpEx+ 训练与推理。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cocktail_party.audio import read_audio_mono, write_wav
from cocktail_party.config import ExperimentConfig
from cocktail_party.dataset import CocktailDataset, get_speaker_splits, load_speaker_splits, rows_for_split
from cocktail_party.selection import embed_audio, load_encoder
from cocktail_party.spex import build_spex, load_spex, spex_training_loss
from cocktail_party.train import _collate, _device, _embed_batch, _trainable_params


def _speaker_id_map(rows: list[dict[str, str]]) -> dict[str, int]:
    speakers = sorted({r["target_speaker"] for r in rows})
    return {spk: idx for idx, spk in enumerate(speakers)}


def _eval_spex(
    model: torch.nn.Module,
    encoder,
    loader: DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
    spk_map: dict[str, int],
) -> tuple[float, float]:
    model.eval()
    scores: list[float] = []
    rms_values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            spk = _embed_batch(encoder, batch["enrollment"], device)
            est = model(mixture, spk)
            from cocktail_party.model import si_sdr_loss

            scores.append(-float(si_sdr_loss(est, target).item()))
            rms_values.append(float(est.pow(2).mean().sqrt().item()))
    mean_rms = float(np.mean(rms_values)) if rms_values else 0.0
    if mean_rms < 1e-6:
        print("  WARNING: SpEx+ validation RMS ~ 0 (possible silent collapse)")
    return float(np.mean(scores)) if scores else float("-inf"), mean_rms


def run_train_spex(cfg: ExperimentConfig) -> Path:
    get_speaker_splits(cfg)
    device = _device(cfg)
    encoder = load_encoder(cfg, str(device))
    for p in encoder.parameters():
        p.requires_grad = False

    train_rows = rows_for_split(cfg, "train")
    val_rows = rows_for_split(cfg, "val")
    spk_map = _speaker_id_map(train_rows + val_rows)
    id_by_sample = {
        r["sample_id"]: spk_map[r["target_speaker"]] for r in train_rows + val_rows
    }

    crop_sec = cfg.spex_crop_sec
    batch_size = cfg.spex_batch_size
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

    model = build_spex(cfg, num_speakers=len(spk_map)).to(device)
    trainable = _trainable_params(model)
    spex_lr = cfg.spex_train_lr
    optimizer = torch.optim.Adam(trainable, lr=spex_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train_epochs, eta_min=1e-5
    )

    ckpt_dir = cfg.spex_checkpoint_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.spex_checkpoint_path
    best_val = float("-inf")
    patience_left = cfg.train_patience

    for epoch in range(1, cfg.train_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"spex train epoch {epoch}"):
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            spk = _embed_batch(encoder, batch["enrollment"], device)
            spk_ids = torch.tensor(
                [id_by_sample[sid] for sid in batch["sample_id"]],
                device=device,
                dtype=torch.long,
            )

            optimizer.zero_grad(set_to_none=True)
            loss = spex_training_loss(model, mixture, target, spk, cfg, spk_ids)
            if not torch.isfinite(loss):
                print("  skip batch: non-finite SpEx+ loss")
                continue
            loss.backward()
            if cfg.train_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.train_grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_si, val_rms = _eval_spex(model, encoder, val_loader, device, cfg, spk_map)
        scheduler.step()
        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        print(
            f"spex epoch {epoch}: train_loss={mean_train:.3f} val_si_sdr={val_si:.3f} "
            f"val_rms={val_rms:.4f}"
        )
        collapsed = val_rms < 1e-5
        if val_si > best_val and not collapsed:
            best_val = val_si
            patience_left = cfg.train_patience
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_si_sdr": val_si, "val_rms": val_rms},
                best_path,
            )
            print(f"  saved SpEx+ -> {best_path}")
        elif val_si > best_val and collapsed:
            print(f"  skip SpEx+ checkpoint: collapse (val_rms={val_rms:.2e})")
            patience_left -= 1
        else:
            patience_left -= 1
        if patience_left <= 0:
            print("SpEx+ early stopping")
            break

    if not best_path.is_file():
        torch.save({"model": model.state_dict()}, best_path)
    return best_path


def run_inference_spex(cfg: ExperimentConfig, rows: list[dict[str, str]] | None = None) -> Path:
    if rows is None:
        rows = rows_for_split(cfg, "test")
    device = _device(cfg)
    encoder = load_encoder(cfg, str(device))
    splits = load_speaker_splits(cfg)
    all_rows = rows_for_split(cfg, "train") + rows_for_split(cfg, "val")
    spk_map_inf = _speaker_id_map(all_rows)
    model = load_spex(cfg, device, num_speakers=len(spk_map_inf))
    output_root = cfg.spex_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    for row in tqdm(rows, desc="inference (SpEx+)"):
        mixture, _ = read_audio_mono(Path(row["mixture_path"]), cfg.sample_rate)
        enrollment, _ = read_audio_mono(Path(row["enrollment_path"]), cfg.sample_rate)
        spk = embed_audio(encoder, enrollment, str(device)).to(device)
        mix_t = torch.from_numpy(mixture).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(mix_t, spk.unsqueeze(0)).squeeze(0).cpu().numpy()
        write_wav(output_root / f"{row['sample_id']}.wav", out, cfg.sample_rate)
    return output_root
