# Cocktail Party — 目标说话人提取对比实验

在鸡尾酒会混合数据上对比 **5 条 TSE 方案**：

| 角色 | 模型 |
|------|------|
| **成功标志** | WeSep-BSRNN(微调) — 域内 SI-SDR 微调 |
| 零样本对照 | WeSep-BSRNN(预训练) |
| 折中/上界 | Oracle+MossFormer2 |
| 自研失败 | AttentionTSE（塌缩）、SpEx+ 自训练 |

## 流程

```text
干净语音 + 噪声 → 混合数据 (12000 条)
    → 训练 AttentionTSE / SpEx+ / WeSep 微调（可选）
    → 测试集 MossFormer2 分离（Oracle 上界）
    → 测试集 5 路 TSE 推理与 SI-SDR 评估
```

## 仓库结构

```text
cocktail/
  run.py
  configs/experiment_config.json
  cocktail_party/
    model.py              # AttentionTSE
    spex.py / spex_train.py
    pretrained_tse.py     # WeSep 预训练/微调推理
    wesep_finetune.py     # WeSep BSRNN 域内微调
    pipeline.py
  checkpoints/            # 不入 Git，见 docs/checkpoints.md
  outputs/                # 不入 Git，运行后生成
  data/                   # 语料与混合数据（raw/ 不入 Git）
  scripts/                # 数据下载与批量运行脚本
  docs/                   # 实验报告与架构说明
```

## Installation

**环境**：Python 3.10+，推荐 Conda + NVIDIA GPU。

```bash
# 1. 克隆并进入仓库
git clone https://github.com/SilasLin/cocktail-party-tse.git
cd cocktail-party-tse

# 2. 创建环境
conda create -n cocktail python=3.10 -y
conda activate cocktail

# 3. 安装 PyTorch（按 CUDA 版本选择，示例 CUDA 12.6）
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 安装 WeSep（目标说话人提取，PyPI 无包）
./scripts/install_wesep.sh

# 6. 配置文件（首次使用）
cp configs/experiment_config.example.json configs/experiment_config.json
# 按需修改 device、work_root 等
```

**模型权重**（约 1.1 GB，不入 Git）需自行下载，见 [docs/checkpoints.md](docs/checkpoints.md)。

**数据集**（LibriSpeech、AISHELL、MUSAN 等，体积大）：

```bash
WORK_ROOT=./data ./scripts/download_datasets.sh
```

## 快速开始

默认 `device: cuda:0`，数据目录 `./data`（可通过 `--work-root` 覆盖）：

```bash
conda activate cocktail
python run.py mix --work-root ./data
python run.py train_wesep --work-root ./data   # WeSep 域内微调
python run.py separate --work-root ./data      # Oracle 需要
python run.py eval --work-root ./data          # 5 路对比
```

可选自研对照：

```bash
python run.py train --work-root ./data         # AttentionTSE
python run.py train_spex --work-root ./data    # SpEx+
```

指定 GPU：`GPU_ID=1 ./scripts/run_eval.sh` 或 `python run.py eval --device cuda:1`。

## 测试集结果（头条）

| 模型 | 整体 mean ΔSI-SDR | 2 说话人 mean ΔSI-SDR |
|------|-------------------|------------------------|
| **WeSep-BSRNN(微调)** | **+11.63 dB** | **+13.58 dB** |
| WeSep-BSRNN(预训练) | +5.32 dB | +8.78 dB |
| Oracle+MossFormer2 | +6.54 dB | — |
| SpEx+ 自训练 | +0.83 dB | — |
| AttentionTSE（塌缩） | 不可比 | — |

完整结果快照：[docs/results/metrics_all_summary.csv](docs/results/metrics_all_summary.csv)

## 输出

| 文件 | 内容 |
|------|------|
| `outputs/metrics_all_summary.csv` | 5 路模型汇总 |
| `outputs/metrics_tse_pretrained_tse_ft_summary.csv` | WeSep 微调分组指标 |
| `outputs/target_extracted/pretrained_tse_ft/` | WeSep 微调输出 wav |

## 文档

| 文档 | 说明 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 模块职责与数据流 |
| [docs/ai_context.md](docs/ai_context.md) | 给后续 AI 的项目记忆（优先阅读） |
| [docs/experiment_report.md](docs/experiment_report.md) | 正式实验报告 |
| [docs/outputs_index.md](docs/outputs_index.md) | 输出产物与 CSV 索引 |
| [docs/experiment_log.md](docs/experiment_log.md) | 数值速查 |
| [docs/checkpoints.md](docs/checkpoints.md) | 权重下载 |

完整文档目录见 [docs/README.md](docs/README.md)。

## Citation

若本仓库对您的研究有帮助，请引用：

```bibtex
@misc{cocktail-party-tse2026,
  title  = {Cocktail Party Target Speaker Extraction Comparison},
  author = {SilasLin},
  year   = {2026},
  url    = {https://github.com/SilasLin/cocktail-party-tse}
}
```

## License

[MIT](LICENSE)
