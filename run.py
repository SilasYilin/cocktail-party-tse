#!/usr/bin/env python3
"""鸡尾酒会目标说话人提取（方案 B：AttentionTSE）统一入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment_config.json"


def _lock_cuda_device(argv: list[str] | None) -> str | None:
    """在 import torch 之前根据配置/参数锁定可见 GPU（仅暴露指定物理卡）。"""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None)
    ns, _ = parser.parse_known_args(argv)

    device = ns.device
    if device is None and ns.config.is_file():
        with ns.config.open(encoding="utf-8") as f:
            device = json.load(f).get("device")

    if device and str(device).startswith("cuda:"):
        gpu_id = str(device).split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return gpu_id
    return None


def _normalize_device_after_lock(device: str) -> str:
    """CUDA_VISIBLE_DEVICES 已掩码时，进程内仅见 cuda:0，对应所选物理卡。"""
    if device.startswith("cuda:") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0"
    return device


# 必须在 cocktail_party（会间接 import torch）之前执行
_LOCKED_GPU = _lock_cuda_device(sys.argv[1:])

from cocktail_party.config import ExperimentConfig
from cocktail_party.pipeline import STAGES, run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="鸡尾酒会目标说话人提取：AttentionTSE 训练与评估流水线"
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=STAGES,
        help=(
            "运行阶段: mix | train | train_spex | train_wesep | separate | eval | eval_pretrained_tse | all\n"
            "  mix                 = 生成混合数据并划分 train/val/test\n"
            "  train               = 训练 AttentionTSE\n"
            "  train_spex          = 训练 SpEx+ 端到端 TSE（自训练对照）\n"
            "  train_wesep         = 微调 WeSep BSRNN（域内 SI-SDR）\n"
            "  separate            = 测试集 MossFormer2 分离（Oracle 上界）\n"
            "  eval                = 测试集 5 路 TSE 对比评估\n"
            "  eval_pretrained_tse = 仅预训练 WeSep BSRNN 推理与指标\n"
            "  all                 = 以上全部"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="实验配置文件 (JSON)")
    parser.add_argument("--work-root", type=Path, default=None, help="覆盖配置中的 work_root（仅数据）")
    parser.add_argument(
        "--project-root", type=Path, default=None, help="覆盖 project_root（checkpoints/outputs）"
    )
    parser.add_argument("--max-samples", type=int, default=None, help="限制推理样本数，0 表示全部")
    parser.add_argument("--device", default=None, help='覆盖 device，例如 cuda:0（默认用配置文件中的 device）')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        print(f"配置文件不存在: {args.config}", file=sys.stderr)
        return 1

    cfg = ExperimentConfig.from_json(args.config)
    if args.work_root is not None:
        cfg.work_root = args.work_root.resolve()
    if args.project_root is not None:
        cfg.project_root = args.project_root.resolve()
    if cfg.project_root is None:
        cfg.project_root = args.config.resolve().parent.parent
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.device is not None:
        cfg.device = args.device
    cfg.device = _normalize_device_after_lock(cfg.device)

    print(f"work_root (数据): {cfg.work_root}")
    print(f"project_root (权重/输出): {cfg.project_root}")
    print(f"checkpoints: {cfg.checkpoint_root}")
    print(f"outputs: {cfg.output_root}")
    if _LOCKED_GPU is not None:
        print(f"CUDA_VISIBLE_DEVICES={_LOCKED_GPU} (进程内 device={cfg.device})")
    else:
        print(f"device: {cfg.device}")
    print(f"stage: {args.stage}")
    results = run_pipeline(cfg, args.stage)
    for name, path in results.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
