"""AttentionTSE 训练与测试集推理。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cocktail_party.audio import read_audio_mono, write_wav
from cocktail_party.config import ExperimentConfig
from cocktail_party.dataset import CocktailDataset, get_speaker_splits, rows_for_split
from cocktail_party.model import (
    build_attention_tse,
    checkpoint_state_dict,
    load_attention_tse,
    mag_l1_loss,
    si_sdr_loss,
)
from cocktail_party.selection import embed_audio, load_encoder


def _device(cfg: ExperimentConfig) -> torch.device:
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        if ":" in cfg.device:
            return torch.device(cfg.device)
        raise ValueError(
            'device 须指定 GPU 编号，例如 "cuda:0"（勿使用裸 "cuda"，以免落到 0 号卡）'
        )
    return torch.device("cpu")


def _trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "mixture": torch.stack([b["mixture"] for b in batch]),
        "enrollment": torch.stack([b["enrollment"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
    }


@torch.no_grad()
def _embed_batch(encoder, enrollment: torch.Tensor, device: torch.device) -> torch.Tensor:
    embs = []
    for i in range(enrollment.size(0)):
        emb = embed_audio(encoder, enrollment[i].cpu().numpy(), str(device))
        embs.append(emb)
    return torch.stack(embs, dim=0).to(device)


def _eval_si_sdr(
    model: torch.nn.Module,
    encoder,
    loader: DataLoader,
    device: torch.device,
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
            loss = si_sdr_loss(est, target)
            scores.append(-float(loss.item()))
            rms_values.append(float(est.pow(2).mean().sqrt().item()))
    mean_rms = float(np.mean(rms_values)) if rms_values else 0.0
    if mean_rms < 1e-6:
        print("  WARNING: validation output RMS ~ 0 (possible silent collapse)")
    mean_si = float(np.mean(scores)) if scores else float("-inf")
    return mean_si, mean_rms


def _training_loss(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    target: torch.Tensor,
    spk: torch.Tensor,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    est = model(mixture, spk)
    loss_si = si_sdr_loss(est, target)
    loss = loss_si
    if cfg.tse_mag_loss_weight > 0:
        loss = loss + cfg.tse_mag_loss_weight * mag_l1_loss(model, est, target)
    if cfg.tse_energy_loss_weight > 0:
        est_rms = est.pow(2).mean(dim=-1).sqrt().clamp(min=1e-8)
        loss = loss + cfg.tse_energy_loss_weight * (-torch.log(est_rms).mean())
    return loss


def run_train(cfg: ExperimentConfig) -> Path:
    """训练 AttentionTSE，保存 best.pt。"""
    get_speaker_splits(cfg)
    device = _device(cfg)
    encoder = load_encoder(cfg, str(device))
    for p in encoder.parameters():
        p.requires_grad = False

    train_rows = rows_for_split(cfg, "train")
    val_rows = rows_for_split(cfg, "val")
    train_ds = CocktailDataset(train_rows, cfg, augment=True)
    val_ds = CocktailDataset(val_rows, cfg, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        drop_last=len(train_ds) >= cfg.train_batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    model = build_attention_tse(cfg).to(device)
    trainable = _trainable_params(model)
    optimizer = torch.optim.Adam(trainable, lr=cfg.train_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train_epochs, eta_min=1e-5
    )

    ckpt_dir = cfg.tse_checkpoint_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.tse_checkpoint_path
    best_val = float("-inf")
    patience_left = cfg.train_patience

    for epoch in range(1, cfg.train_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            spk = _embed_batch(encoder, batch["enrollment"], device)

            optimizer.zero_grad(set_to_none=True)
            loss = _training_loss(model, mixture, target, spk, cfg)
            loss.backward()
            if cfg.train_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.train_grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_si, val_rms = _eval_si_sdr(model, encoder, val_loader, device)
        scheduler.step()
        mean_train = float(np.mean(train_losses)) if train_losses else 0.0
        print(
            f"epoch {epoch}: train_loss={mean_train:.3f} val_si_sdr={val_si:.3f} "
            f"val_rms={val_rms:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        collapsed = val_rms < 1e-5
        if val_si > best_val and not collapsed:
            best_val = val_si
            patience_left = cfg.train_patience
            torch.save(
                {
                    "model": checkpoint_state_dict(model),
                    "epoch": epoch,
                    "val_si_sdr": val_si,
                    "val_rms": val_rms,
                },
                best_path,
            )
            print(f"  saved checkpoint -> {best_path} (val_si_sdr={val_si:.3f})")
        elif val_si > best_val and collapsed:
            print(f"  skip checkpoint: silent collapse (val_rms={val_rms:.2e})")
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping")
                break

    if not best_path.is_file():
        torch.save({"model": checkpoint_state_dict(model)}, best_path)
    return best_path


def run_inference_attention_tse(cfg: ExperimentConfig, rows: list[dict[str, str]] | None = None) -> Path:
    """在指定样本上运行 AttentionTSE 推理（默认测试集）。"""
    if rows is None:
        rows = rows_for_split(cfg, "test")
    device = _device(cfg)
    encoder = load_encoder(cfg, str(device))
    model = load_attention_tse(cfg, device)

    output_root = cfg.tse_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for row in tqdm(rows, desc="inference (AttentionTSE)"):
        mixture, _ = read_audio_mono(Path(row["mixture_path"]), cfg.sample_rate)
        enrollment, _ = read_audio_mono(Path(row["enrollment_path"]), cfg.sample_rate)
        spk = embed_audio(encoder, enrollment, str(device)).to(device)
        mix_t = torch.from_numpy(mixture).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(mix_t, spk.unsqueeze(0)).squeeze(0).cpu().numpy()
        write_wav(output_root / f"{row['sample_id']}.wav", out, cfg.sample_rate)

    return output_root
