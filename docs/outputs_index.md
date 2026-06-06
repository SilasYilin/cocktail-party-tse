# 实验输出索引

> 供后续 AI / 人工快速定位产物。路径均相对于 **project_root**（仓库根目录，默认与 `run.py` 同级）。  
> 数据根目录 **work_root** 默认为 `./data`，可通过 `--work-root` 或配置文件覆盖。

## 1. 目录总览

```text
outputs/
├── metrics_all_detail.csv          # 5 路 TSE + Oracle 逐条指标（8960 行 = 1792×5）
├── metrics_all_summary.csv         # 5 路分组汇总（60 行）
├── metrics_tse_*_detail.csv        # 各模型逐条
├── metrics_tse_*_summary.csv       # 各模型按 SNR×说话人数分组
├── metrics_separation_mossformer2_*.csv   # 盲分离指标（Oracle 上游）
├── speaker_splits.json             # train/val/test 说话人划分
├── run_*.log                       # 各阶段运行日志
├── separated/mossformer2/          # MossFormer2 分离 wav（Oracle 用）
└── target_extracted/
    ├── pretrained_tse/             # WeSep 预训练输出（1792 wav，~524MB）
    ├── pretrained_tse_ft/          # WeSep 微调输出（1792 wav，~524MB）
    └── spex_plus/                  # 自训练 SpEx+ 输出（1792 wav，~524MB）

checkpoints/
├── wesep_bsrnn/avg_model.pt        # WeSep 预训练（~270MB）
├── wesep_bsrnn_ft/best.pt          # WeSep 微调（~106MB，epoch 4 best）
├── clearvoice/                     # MossFormer2_SS_16K
├── spex_plus/best.pt               # 自训练 SpEx+ epoch 2
├── attention_tse/best.pt           # AttentionTSE 塌缩前 checkpoint
└── spkrec-ecapa-voxceleb/          # ECAPA（训练用，已不参与评测）
```

**注意**：`outputs/target_extracted/attention_tse/` 的 wav 已清理以省空间；`metrics_tse_attention_tse_*.csv` 仍保留。

## 2. 指标 CSV 说明

### 2.1 合并表

| 文件 | 行数 | 内容 |
|------|------|------|
| `metrics_all_detail.csv` | 8960 | 5 模型 × 1792 测试样本 |
| `metrics_all_summary.csv` | 60 | 5 模型 × 12 分组（4 SNR × 3 说话人数） |

### 2.2 各模型明细

| 文件前缀 | 模型 | 测试集整体 mean ΔSI-SDR | mean SI-SDR_out |
|----------|------|-------------------------|-----------------|
| `metrics_tse_pretrained_tse_ft` | **WeSep-BSRNN(微调)** | **+11.630** | **+5.451** |
| `metrics_tse_pretrained_tse` | WeSep-BSRNN(预训练) | +5.320 | -0.860 |
| `metrics_tse_oracle` | Oracle+MossFormer2 | +6.542 | +0.363 |
| `metrics_tse_attention_tse` | AttentionTSE（塌缩） | +6.164 * | -0.015 |
| `metrics_tse_spex` | SpEx+ 自训练 | +0.830 | -5.349 |

\* AttentionTSE 的 Δ 在输出近零时为假象。

### 2.3 detail 列字段

| 列名 | 含义 |
|------|------|
| `sample_id` | 样本 ID，与 metadata 一致 |
| `task` | `target_extraction` 或 `oracle_tse` |
| `model` | 显示名称 |
| `condition` | `clean` / `5.0` / `0.0` / `-5.0` |
| `n_speakers` | 2 / 3 / 4 |
| `si_sdr_in` | 混合音相对目标的 SI-SDR |
| `si_sdr_out` | 估计音相对目标的 SI-SDR |
| `delta_si_sdr` | out − in |
| `estimate_paths` | 估计 wav 绝对路径 |

### 2.4 已删除 / 不再维护

| 路径 | 原因 |
|------|------|
| `metrics_tse_ecapa_*.csv` | ECAPA+MossFormer2 已从对比方案移除 |
| `metrics_tse_pretrained_spex_*.csv` | ClearerVoice 8k 预训练已弃用 |
| `target_extracted/ecapa_mossformer2/` | 同上 |
| `target_extracted/pretrained_spex/` | 同上 |

## 3. 音频输出

| 目录 | 条数 | 采样率 | 生成命令 |
|------|------|--------|----------|
| `target_extracted/pretrained_tse/` | 1792 | 16 kHz | `eval` / `eval_pretrained_tse` |
| `target_extracted/pretrained_tse_ft/` | 1792 | 16 kHz | `eval`（需 `wesep_bsrnn_ft/best.pt`） |
| `target_extracted/spex_plus/` | 1792 | 16 kHz | `eval`（需 `spex_plus/best.pt`） |
| `separated/mossformer2/` | 1792×N | 16 kHz | `separate` |

命名：`{sample_id}.wav`，与 `metadata.csv` 的 `sample_id` 对应。

## 4. 运行日志

| 日志 | 阶段 | 备注 |
|------|------|------|
| `run_mix.log` | 混合数据生成 | |
| `run_train.log` | AttentionTSE 训练 | 塌缩现象见报告 |
| `run_train_spex.log` | SpEx+ 训练 | epoch 2 best |
| `run_train_wesep.log` | WeSep 微调 | epoch 1–4，best=epoch 4 |
| `run_separate.log` | MossFormer2 分离 | |
| `run_eval.log` | 多路评测 | |
| `run_eval_pretrained_tse.log` | 仅预训练 WeSep 评测 | |

## 5. 快速查询命令

```bash
# 整体均值
python -c "
import pandas as pd
d=pd.read_csv('outputs/metrics_tse_pretrained_tse_ft_detail.csv')
print(d['delta_si_sdr'].mean(), d['si_sdr_out'].mean())
"

# 2 说话人分组
python -c "
import pandas as pd
s=pd.read_csv('outputs/metrics_tse_pretrained_tse_ft_summary.csv')
print(s[s.n_speakers==2][['condition','mean_delta_si_sdr']])
"

# 重建 metrics_all（不重新推理）
python -c "
from pathlib import Path
from cocktail_party.config import ExperimentConfig
from cocktail_party.pipeline import rebuild_metrics_all
cfg=ExperimentConfig.from_json('configs/experiment_config.json')
# work_root / project_root 默认从配置解析；若数据在外部磁盘可显式设置：
# cfg.work_root=Path('/path/to/cocktail-data')
print(rebuild_metrics_all(cfg))
"
```

## 6. 相关文档

| 文档 | 用途 |
|------|------|
| [ai_context.md](ai_context.md) | 给后续 AI 的项目记忆与决策脉络 |
| [experiment_report.md](experiment_report.md) | 正式实验报告 |
| [experiment_log.md](experiment_log.md) | 精简实验记录与数值 |
| [checkpoints.md](checkpoints.md) | 权重下载与校验 |
