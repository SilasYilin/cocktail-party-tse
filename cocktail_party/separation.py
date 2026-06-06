"""MossFormer2 盲分离（ClearVoice）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from cocktail_party.config import ExperimentConfig
from cocktail_party.io import read_metadata


def _patch_clearvoice_gpu_selection(cfg: ExperimentConfig) -> None:
    import torch
    from clearvoice.networks import SpeechModel

    if getattr(SpeechModel.get_free_gpu, "__cocktail_patched__", False):
        return

    def get_free_gpu_visible():
        if cfg.device == "cpu" or not torch.cuda.is_available():
            return None
        count = torch.cuda.device_count()
        if count == 0:
            return None
        if cfg.device.startswith("cuda:"):
            idx = int(cfg.device.split(":", 1)[1])
            if idx < 0 or idx >= count:
                raise RuntimeError(
                    f"device={cfg.device!r} 超出当前可见 GPU 数量 ({count})。"
                    "若设置了 CUDA_VISIBLE_DEVICES，请使用 cuda:0。"
                )
            return idx
        best_idx = 0
        best_free = -1
        for i in range(count):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_idx = i
        return best_idx

    get_free_gpu_visible.__cocktail_patched__ = True
    SpeechModel.get_free_gpu = get_free_gpu_visible


def write_clearvoice_output(engine, output, output_dir: Path, sample_rate: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        engine.write(output, output_path=str(output_dir))
        return
    except Exception:
        pass
    if isinstance(output, list):
        for idx, spk in enumerate(output):
            audio = np.squeeze(np.asarray(spk)).astype(np.float32)
            sf.write(str(output_dir / f"source{idx + 1}.wav"), audio, sample_rate)
        return
    arr = np.squeeze(np.asarray(output))
    if arr.ndim == 1:
        sf.write(str(output_dir / "source1.wav"), arr.astype(np.float32), sample_rate)
        return
    if arr.shape[0] < arr.shape[-1]:
        arr = arr.T
    for idx in range(arr.shape[1]):
        sf.write(
            str(output_dir / f"source{idx + 1}.wav"),
            arr[:, idx].astype(np.float32),
            sample_rate,
        )


def create_separation_engine(cfg: ExperimentConfig):
    import torch
    from clearvoice.network_wrapper import network_wrapper
    from clearvoice.networks import CLS_MossFormer2_SS_16K

    wrapper = network_wrapper()
    wrapper.model_name = cfg.separation_model
    wrapper.load_args_ss()
    wrapper.args.checkpoint_dir = str(cfg.separation_checkpoint_path)
    wrapper.args.use_cuda = 1 if cfg.device.startswith("cuda") and torch.cuda.is_available() else 0
    wrapper.args.task = "speech_separation"
    wrapper.args.network = cfg.separation_model
    _patch_clearvoice_gpu_selection(cfg)
    return CLS_MossFormer2_SS_16K(wrapper.args)


def run_separation(cfg: ExperimentConfig, rows: list[dict[str, str]] | None = None) -> Path:
    ckpt = cfg.separation_checkpoint_path
    marker = ckpt / "last_best_checkpoint"
    if not marker.is_file():
        raise FileNotFoundError(
            f"未找到分离模型权重：{marker}。请将 MossFormer2 权重放在 {ckpt}（含 last_best_checkpoint 与 .pt 文件）。"
        )
    engine = create_separation_engine(cfg)
    if rows is None:
        rows = read_metadata(cfg.metadata_path)
    if cfg.max_samples > 0:
        rows = rows[: cfg.max_samples]
    output_root = cfg.separation_output_root
    for row in tqdm(rows, desc="separation"):
        output_dir = output_root / row["sample_id"]
        if output_dir.exists() and list(output_dir.glob("*.wav")):
            continue
        out = engine.process(input_path=row["mixture_path"], online_write=False)
        write_clearvoice_output(engine, out, output_dir, cfg.sample_rate)
    return output_root
