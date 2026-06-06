"""方案 B 流水线：混合数据 → 训练 → 测试集多路 TSE 评估。"""
from __future__ import annotations

import csv
from pathlib import Path

from cocktail_party.config import ExperimentConfig
from cocktail_party.dataset import get_speaker_splits, rows_for_split, speaker_splits_path
from cocktail_party.metrics import compute_metrics
from cocktail_party.mixtures import generate_mixtures
from cocktail_party.separation import run_separation
from cocktail_party.pretrained_tse import (
    run_inference_pretrained_tse,
    run_inference_pretrained_tse_ft,
)
from cocktail_party.spex_train import run_inference_spex, run_train_spex
from cocktail_party.train import run_inference_attention_tse, run_train
from cocktail_party.wesep_finetune import run_train_wesep

STAGES = (
    "mix",
    "train",
    "train_spex",
    "train_wesep",
    "separate",
    "eval",
    "eval_pretrained_tse",
    "all",
)


def _merge_csv_files(paths: list[Path], output_path: Path) -> None:
    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for p in paths:
        if not p.is_file():
            continue
        with p.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and not fieldnames:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                all_rows.append(row)

    if not all_rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def _collect_tse_metric_paths(cfg: ExperimentConfig) -> tuple[list[Path], list[Path]]:
    """返回保留方案的 detail/summary CSV 路径列表。"""
    out = cfg.output_root
    detail_paths: list[Path] = [
        out / "metrics_tse_attention_tse_detail.csv",
        out / "metrics_tse_spex_detail.csv",
        out / "metrics_tse_pretrained_tse_detail.csv",
        out / "metrics_tse_pretrained_tse_ft_detail.csv",
        out / "metrics_tse_oracle_detail.csv",
    ]
    summary_paths: list[Path] = [
        out / "metrics_tse_attention_tse_summary.csv",
        out / "metrics_tse_spex_summary.csv",
        out / "metrics_tse_pretrained_tse_summary.csv",
        out / "metrics_tse_pretrained_tse_ft_summary.csv",
        out / "metrics_tse_oracle_summary.csv",
    ]
    return detail_paths, summary_paths


def run_eval_comparison(cfg: ExperimentConfig, *, include_pretrained: bool = True) -> dict[str, Path]:
    """在测试集上对比保留的 5 条方案。"""
    test_rows = rows_for_split(cfg, "test")
    if not test_rows:
        raise RuntimeError("测试集为空，请先运行 mix 阶段生成数据并划分 speaker_splits")

    sep_root = cfg.separation_output_root
    if not sep_root.is_dir() or not any(sep_root.iterdir()):
        run_separation(cfg, rows=test_rows)

    run_inference_attention_tse(cfg, test_rows)

    if cfg.spex_checkpoint_path.is_file():
        run_inference_spex(cfg, test_rows)

    if include_pretrained and cfg.pretrained_tse_checkpoint_path.is_file():
        run_inference_pretrained_tse(cfg, test_rows)

    if cfg.pretrained_tse_ft_checkpoint_path.is_file():
        run_inference_pretrained_tse_ft(cfg, test_rows)

    results: dict[str, Path] = {}
    detail_paths: list[Path] = []
    summary_paths: list[Path] = []

    tse_configs = [
        (cfg.tse_output_root, cfg.tse_metrics_name, "attention_tse", None),
    ]
    if cfg.spex_output_root.is_dir() and any(cfg.spex_output_root.glob("*.wav")):
        tse_configs.append((cfg.spex_output_root, cfg.spex_metrics_name, "spex", None))
    if include_pretrained and cfg.pretrained_tse_output_root.is_dir() and any(
        cfg.pretrained_tse_output_root.glob("*.wav")
    ):
        tse_configs.append(
            (cfg.pretrained_tse_output_root, cfg.pretrained_tse_metrics_name, "pretrained_tse", None)
        )
    if cfg.pretrained_tse_ft_output_root.is_dir() and any(
        cfg.pretrained_tse_ft_output_root.glob("*.wav")
    ):
        tse_configs.append(
            (
                cfg.pretrained_tse_ft_output_root,
                cfg.pretrained_tse_ft_metrics_name,
                "pretrained_tse_ft",
                None,
            )
        )

    for est_root, model_name, tag, sep_root_opt in tse_configs:
        d, s = compute_metrics(
            cfg,
            task="target_extraction",
            estimate_root=est_root,
            model_name=model_name,
            detail_name=f"metrics_tse_{tag}_detail.csv",
            summary_name=f"metrics_tse_{tag}_summary.csv",
            separation_root=sep_root_opt,
            rows=test_rows,
        )
        detail_paths.append(d)
        summary_paths.append(s)
        results[f"metrics_tse_{tag}_detail"] = d
        results[f"metrics_tse_{tag}_summary"] = s

    d, s = compute_metrics(
        cfg,
        task="oracle_tse",
        estimate_root=sep_root,
        model_name=cfg.oracle_metrics_name,
        detail_name="metrics_tse_oracle_detail.csv",
        summary_name="metrics_tse_oracle_summary.csv",
        rows=test_rows,
    )
    detail_paths.append(d)
    summary_paths.append(s)
    results["metrics_tse_oracle_detail"] = d
    results["metrics_tse_oracle_summary"] = s

    all_detail = cfg.output_root / "metrics_all_detail.csv"
    all_summary = cfg.output_root / "metrics_all_summary.csv"
    _merge_csv_files(detail_paths, all_detail)
    _merge_csv_files(summary_paths, all_summary)
    results["metrics_all_detail"] = all_detail
    results["metrics_all_summary"] = all_summary
    return results


def run_stage(cfg: ExperimentConfig, stage: str) -> dict[str, Path]:
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"未知阶段: {stage}，可选: {', '.join(STAGES)}")

    results: dict[str, Path] = {}
    if stage in ("mix", "all"):
        results["metadata"] = generate_mixtures(cfg)
        get_speaker_splits(cfg)
        results["speaker_splits"] = speaker_splits_path(cfg)

    if stage in ("train", "all"):
        if not speaker_splits_path(cfg).is_file():
            get_speaker_splits(cfg)
        results["checkpoint"] = run_train(cfg)

    if stage in ("train_spex", "all"):
        if not speaker_splits_path(cfg).is_file():
            get_speaker_splits(cfg)
        results["spex_checkpoint"] = run_train_spex(cfg)

    if stage in ("train_wesep", "all"):
        if not speaker_splits_path(cfg).is_file():
            get_speaker_splits(cfg)
        results["wesep_ft_checkpoint"] = run_train_wesep(cfg)

    if stage in ("separate", "all"):
        test_rows = rows_for_split(cfg, "test")
        results["separation"] = run_separation(cfg, rows=test_rows)

    if stage in ("eval", "all"):
        results.update(run_eval_comparison(cfg))

    if stage == "eval_pretrained_tse":
        test_rows = rows_for_split(cfg, "test")
        if not test_rows:
            raise RuntimeError("测试集为空，请先运行 mix 阶段")
        results["pretrained_tse_output"] = run_inference_pretrained_tse(cfg, test_rows)
        d, s = compute_metrics(
            cfg,
            task="target_extraction",
            estimate_root=cfg.pretrained_tse_output_root,
            model_name=cfg.pretrained_tse_metrics_name,
            detail_name="metrics_tse_pretrained_tse_detail.csv",
            summary_name="metrics_tse_pretrained_tse_summary.csv",
            rows=test_rows,
        )
        results["metrics_tse_pretrained_tse_detail"] = d
        results["metrics_tse_pretrained_tse_summary"] = s

    return results


def run_pipeline(cfg: ExperimentConfig, stage: str = "all") -> dict[str, Path]:
    return run_stage(cfg, stage)


def rebuild_metrics_all(cfg: ExperimentConfig) -> dict[str, Path]:
    """从已有 per-model CSV 重建 metrics_all（不重新推理）。"""
    detail_paths, summary_paths = _collect_tse_metric_paths(cfg)
    all_detail = cfg.output_root / "metrics_all_detail.csv"
    all_summary = cfg.output_root / "metrics_all_summary.csv"
    _merge_csv_files(detail_paths, all_detail)
    _merge_csv_files(summary_paths, all_summary)
    return {"metrics_all_detail": all_detail, "metrics_all_summary": all_summary}
