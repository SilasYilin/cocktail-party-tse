"""分离与目标说话人提取指标计算。"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cocktail_party.audio import align_length, read_audio_mono, si_sdr
from cocktail_party.config import ExperimentConfig
from cocktail_party.io import find_separation_estimates, find_target_estimate, read_metadata

try:
    import mir_eval.separation as mir_eval_sep

    mir_eval = mir_eval_sep
except Exception:
    mir_eval = None

try:
    from pesq import pesq as _pesq_fn

    _HAS_PESQ = True
except Exception:
    _HAS_PESQ = False


def bss_eval(
    references: list[np.ndarray], estimates: list[np.ndarray]
) -> tuple[float, float, float] | tuple[None, None, None]:
    if mir_eval is None:
        return None, None, None
    length = min(min(x.size for x in references), min(x.size for x in estimates))
    refs = np.stack([x[:length] for x in references], axis=0)
    ests = np.stack([x[:length] for x in estimates], axis=0)
    sdr, sir, sar, _ = mir_eval.bss_eval_sources(refs, ests, compute_permutation=False)
    return float(np.mean(sdr)), float(np.mean(sir)), float(np.mean(sar))


def compute_pesq(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> float | None:
    if not _HAS_PESQ:
        return None
    ref, est = align_length([reference, estimate])
    mode = "wb" if sample_rate >= 16000 else "nb"
    try:
        return float(_pesq_fn(sample_rate, ref, est, mode))
    except Exception:
        return None


def metric_row_base(row: dict[str, str], model_name: str, task: str) -> dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "task": task,
        "model": model_name,
        "dataset": row.get("dataset", "mixed"),
        "condition": row.get("condition", ""),
        "n_speakers": row.get("n_speakers", ""),
    }


def evaluate_separation(
    row: dict[str, str],
    estimate_root: Path,
    model_name: str,
    sample_rate: int,
) -> dict[str, str] | None:
    sep_paths = find_separation_estimates(estimate_root, row["sample_id"])
    if not sep_paths:
        return None
    ref_paths = [Path(row["target_source_path"])]
    for p in row.get("source_paths", row.get("other_source_paths", "")).split(";"):
        p = p.strip()
        if p and Path(p) != Path(row["target_source_path"]):
            ref_paths.append(Path(p))
    refs = [read_audio_mono(p, sample_rate)[0] for p in ref_paths]
    ests = [read_audio_mono(p, sample_rate)[0] for p in sep_paths]
    sdr, sir, sar = bss_eval(refs, ests)
    out = metric_row_base(row, model_name, "separation")
    out.update(
        {
            "si_sdr_in": "",
            "si_sdr_out": "",
            "delta_si_sdr": "",
            "sdr": "" if sdr is None else f"{sdr:.3f}",
            "sir": "" if sir is None else f"{sir:.3f}",
            "sar": "" if sar is None else f"{sar:.3f}",
            "pesq": "",
            "selection_correct": "",
            "oracle_si_sdr_out": "",
            "oracle_delta_si_sdr": "",
            "estimate_paths": ";".join(str(p.resolve()) for p in sep_paths),
        }
    )
    return out


def oracle_select(
    estimate_paths: list[Path], target_path: Path, sample_rate: int
) -> tuple[int, float]:
    ref, _ = read_audio_mono(target_path, sample_rate)
    scores: list[float] = []
    for path in estimate_paths:
        est, _ = read_audio_mono(path, sample_rate)
        ref_a, est_a = align_length([ref, est])
        scores.append(si_sdr(est_a, ref_a))
    best_idx = int(np.argmax(scores))
    return best_idx, float(scores[best_idx])


def evaluate_tse(
    row: dict[str, str],
    estimate_root: Path,
    model_name: str,
    sample_rate: int,
    separation_root: Path | None = None,
    compute_pesq_flag: bool = False,
) -> dict[str, str] | None:
    estimate_path = find_target_estimate(estimate_root, row["sample_id"])
    if estimate_path is None:
        return None
    ref, _ = read_audio_mono(Path(row["target_source_path"]), sample_rate)
    est, _ = read_audio_mono(estimate_path, sample_rate)
    mix, _ = read_audio_mono(Path(row["mixture_path"]), sample_rate)
    ref, est, mix = align_length([ref, est, mix])
    in_si_sdr = si_sdr(mix, ref)
    out_si_sdr = si_sdr(est, ref)

    oracle_out = ""
    oracle_delta = ""
    selection_correct = ""
    if separation_root is not None:
        sep_paths = find_separation_estimates(separation_root, row["sample_id"])
        if sep_paths:
            best_idx, best_score = oracle_select(
                sep_paths, Path(row["target_source_path"]), sample_rate
            )
            oracle_out = f"{best_score:.3f}"
            oracle_delta = f"{best_score - in_si_sdr:.3f}"
            selected_audio, _ = read_audio_mono(estimate_path, sample_rate)
            match_scores: list[float] = []
            for sp in sep_paths:
                cand, _ = read_audio_mono(sp, sample_rate)
                n = min(selected_audio.size, cand.size)
                overlap = si_sdr(selected_audio[:n], cand[:n])
                match_scores.append(overlap)
            selection_correct = str(int(np.argmax(match_scores) == best_idx))

    pesq_val = ""
    if compute_pesq_flag:
        p = compute_pesq(ref, est, sample_rate)
        if p is not None:
            pesq_val = f"{p:.3f}"

    out = metric_row_base(row, model_name, "target_extraction")
    out.update(
        {
            "si_sdr_in": f"{in_si_sdr:.3f}",
            "si_sdr_out": f"{out_si_sdr:.3f}",
            "delta_si_sdr": f"{out_si_sdr - in_si_sdr:.3f}",
            "sdr": "",
            "sir": "",
            "sar": "",
            "pesq": pesq_val,
            "selection_correct": selection_correct,
            "oracle_si_sdr_out": oracle_out,
            "oracle_delta_si_sdr": oracle_delta,
            "estimate_paths": str(estimate_path.resolve()),
        }
    )
    return out


def evaluate_oracle_tse(
    row: dict[str, str],
    separation_root: Path,
    model_name: str,
    sample_rate: int,
    compute_pesq_flag: bool = False,
) -> dict[str, str] | None:
    sep_paths = find_separation_estimates(separation_root, row["sample_id"])
    if not sep_paths:
        return None
    ref, _ = read_audio_mono(Path(row["target_source_path"]), sample_rate)
    mix, _ = read_audio_mono(Path(row["mixture_path"]), sample_rate)
    ref, mix = align_length([ref, mix])
    in_si_sdr = si_sdr(mix, ref)
    best_idx, best_score = oracle_select(
        sep_paths, Path(row["target_source_path"]), sample_rate
    )
    est, _ = read_audio_mono(sep_paths[best_idx], sample_rate)
    ref_a, est_a = align_length([ref, est])
    out_si_sdr = si_sdr(est_a, ref_a)
    pesq_val = ""
    if compute_pesq_flag:
        p = compute_pesq(ref_a, est_a, sample_rate)
        if p is not None:
            pesq_val = f"{p:.3f}"
    out = metric_row_base(row, model_name, "target_extraction")
    out.update(
        {
            "si_sdr_in": f"{in_si_sdr:.3f}",
            "si_sdr_out": f"{out_si_sdr:.3f}",
            "delta_si_sdr": f"{out_si_sdr - in_si_sdr:.3f}",
            "sdr": "",
            "sir": "",
            "sar": "",
            "pesq": pesq_val,
            "selection_correct": "1",
            "oracle_si_sdr_out": f"{best_score:.3f}",
            "oracle_delta_si_sdr": f"{best_score - in_si_sdr:.3f}",
            "estimate_paths": str(sep_paths[best_idx].resolve()),
        }
    )
    return out


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    groups: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["task"],
            row["model"],
            row["dataset"],
            row["condition"],
            row["n_speakers"],
        )
        groups[key].append(row)

    fieldnames = [
        "task",
        "model",
        "dataset",
        "condition",
        "n_speakers",
        "count",
        "mean_si_sdr_in",
        "mean_si_sdr_out",
        "mean_delta_si_sdr",
        "mean_sdr",
        "mean_sir",
        "mean_sar",
        "mean_pesq",
        "selection_accuracy",
        "mean_oracle_delta_si_sdr",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for key, items in sorted(groups.items()):
            def mean_of(name: str) -> str:
                vals = [float(r[name]) for r in items if r.get(name, "")]
                return f"{np.mean(vals):.3f}" if vals else ""

            sel_vals = [r.get("selection_correct", "") for r in items]
            sel_valid = [float(v) for v in sel_vals if v != ""]
            sel_acc = f"{np.mean(sel_valid):.3f}" if sel_valid else ""
            writer.writerow(
                {
                    "task": key[0],
                    "model": key[1],
                    "dataset": key[2],
                    "condition": key[3],
                    "n_speakers": key[4],
                    "count": str(len(items)),
                    "mean_si_sdr_in": mean_of("si_sdr_in"),
                    "mean_si_sdr_out": mean_of("si_sdr_out"),
                    "mean_delta_si_sdr": mean_of("delta_si_sdr"),
                    "mean_sdr": mean_of("sdr"),
                    "mean_sir": mean_of("sir"),
                    "mean_sar": mean_of("sar"),
                    "mean_pesq": mean_of("pesq"),
                    "selection_accuracy": sel_acc,
                    "mean_oracle_delta_si_sdr": mean_of("oracle_delta_si_sdr"),
                }
            )


def compute_metrics(
    cfg: ExperimentConfig,
    task: str,
    estimate_root: Path,
    model_name: str,
    detail_name: str,
    summary_name: str,
    separation_root: Path | None = None,
    rows: list[dict[str, str]] | None = None,
) -> tuple[Path, Path]:
    if rows is None:
        rows = read_metadata(cfg.metadata_path)
    if cfg.max_samples > 0:
        rows = rows[: cfg.max_samples]

    metric_rows: list[dict[str, str]] = []
    for row in rows:
        if task == "separation":
            m = evaluate_separation(row, estimate_root, model_name, cfg.sample_rate)
        elif task == "oracle_tse":
            sep = separation_root if separation_root is not None else estimate_root
            m = evaluate_oracle_tse(row, sep, model_name, cfg.sample_rate)
        else:
            m = evaluate_tse(
                row,
                estimate_root,
                model_name,
                cfg.sample_rate,
                separation_root=separation_root,
            )
        if m is not None:
            metric_rows.append(m)

    if not metric_rows and estimate_root.is_dir():
        n_sub = sum(1 for _ in estimate_root.iterdir())
        print(
            f"任务 {task} 在 {estimate_root} 下无可用评估样本（子目录约 {n_sub} 个）"
        )

    detail_path = cfg.output_root / detail_name
    summary_path = cfg.output_root / summary_name
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if metric_rows:
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)
        write_summary(metric_rows, summary_path)
    return detail_path, summary_path
