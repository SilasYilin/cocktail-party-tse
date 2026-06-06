# AI 上下文记忆（Cocktail Party TSE 实验）

> **目的**：让后续对话中的 AI 无需重读全仓库即可恢复实验状态。  
> **最后更新**：2026-06-06（WeSep 微调完成、5 路对比定稿）

---

## 1. 项目一句话

在自建的 16 kHz 中英鸡尾酒会混合数据上，对比目标说话人提取（TSE）；**最终成功方案**为 **WeSep BSRNN+ECAPA 域内微调**，测试集整体 **ΔSI-SDR +11.6 dB**，2 说话人 **+13.6 dB**。

## 2. 路径约定

| 名称 | 路径 |
|------|------|
| **project_root**（代码 / 权重 / outputs） | 仓库根目录，或 `configs/experiment_config.json` 中的 `project_root` |
| **work_root**（原始与混合数据） | 默认 `./data`，可用 `--work-root` 覆盖 |
| Conda 环境 | `cocktail`（推荐） |
| GPU | 配置 `device: cuda:0`；`run.py` 会设置 `CUDA_VISIBLE_DEVICES`，进程内通常为 `cuda:0` |

### 开发者备注（本机实验环境）

| 名称 | 路径 |
|------|------|
| project_root | `/home/shenyou/code/yilin/cocktail` |
| work_root | `/data/shenyou/yilin/cocktail` |
| GPU | 物理卡 7（`GPU_ID=7` 或 `device: cuda:7`） |

## 3. 实验演进时间线

1. **AttentionTSE（自研）**：掩码 + ECAPA 条件 → 训练塌缩（输出 RMS≈0），ΔSI-SDR 为假象。
2. **ECAPA+MossFormer2**：两阶段基线，整体 +4.1 dB → **已从最终对比表移除**（编码器仍用于自研模型训练）。
3. **SpEx+ 自训练**：同数据端到端，epoch 2 best 后塌缩，整体仅 +0.8 dB → 保留为**失败样例**。
4. **ClearerVoice SpEx+ 预训练**（WSJ0 8 kHz）：零样本 **-7.9 dB** → 域不匹配，已弃用并删权重。
5. **WeSep BSRNN+ECAPA 预训练**（VoxCeleb1 16 kHz）：零样本 **+5.3 dB** → 保留为对照。
6. **WeSep 域内微调**（冻结 ECAPA，SI-SDR 训分离器，4 epoch）：**+11.6 dB** → **实验成功标志**。

## 4. 当前保留的 5 条方案

| 标签 | 角色 | 整体 ΔSI-SDR | 2spk ΔSI-SDR |
|------|------|--------------|--------------|
| WeSep-BSRNN(微调) | **成功** | +11.630 | +13.584 |
| WeSep-BSRNN(预训练) | 零样本对照 | +5.320 | +8.527 |
| Oracle+MossFormer2 | 上界参考 | +6.542 | +10.497 |
| AttentionTSE | 塌缩失败 | 不可比 | 假象 ~4.0 |
| SpEx+ 自训练 | 欠训练失败 | +0.830 | +0.920 |

## 5. 关键代码入口

| 文件 | 作用 |
|------|------|
| `run.py` | CLI：`mix \| train \| train_spex \| train_wesep \| separate \| eval \| all` |
| `cocktail_party/pipeline.py` | 阶段编排；`eval` 跑 5 路；`rebuild_metrics_all()` 合并 CSV |
| `cocktail_party/pretrained_tse.py` | WeSep 预训练/微调推理 |
| `cocktail_party/wesep_finetune.py` | WeSep 微调训练 |
| `cocktail_party/model.py` | AttentionTSE + `si_sdr_loss` |
| `configs/experiment_config.json` | 全部超参 |

## 6. WeSep 接入要点（环境坑）

- PyPI 无 `wesep`，需 `pip install git+https://github.com/wenet-e2e/WeSep.git`。
- 权重目录：`checkpoints/wesep_bsrnn/{avg_model.pt, config.yaml}`。
- 曾 patch site-packages：`wespeaker/__init__.py` 避免 s3prl 慢导入；`extractor.py` VAD lazy 加载。
- 推理：`load_model_local` → `set_vad(False)` → `extract_speech_from_pcm`；enrollment 为**原始波形**（非 ECAPA 向量）。
- 微调：enrollment 转 kaldi fbank (80 mel) → `model(mix, fbank)` → `si_sdr_loss`；**冻结 `spk_model`**。

## 7. 微调超参（当前 best）

```json
"wesep_ft_lr": 1e-4,
"wesep_ft_epochs": 15,
"wesep_ft_crop_sec": 4.0,
"wesep_ft_batch_size": 4,
"wesep_ft_patience": 5
```

训练日志 `outputs/run_train_wesep.log`：epoch 4 val_si_sdr=4.529（crop 验证集）→ 测试集整体 +11.6 dB。

## 8. 数据集

- 12000 混合，speaker-disjoint 划分：train 8488 / val 1720 / test **1792**
- 条件：2/3/4 说话人 × clean/5/0/-5 dB SNR
- 语料：LibriSpeech、THCHS30、AISHELL、DIY + MUSAN 噪声
- 划分文件：`outputs/speaker_splits.json`

## 9. 评测注意

- **务必看 `si_sdr_out` 和输出 RMS**，不能只看 `delta_si_sdr`（AttentionTSE 塌缩反例）。
- 头条成功指标约定：**2 说话人条件的 mean ΔSI-SDR**（目标 ~10 dB，微调已达 13.6 dB）。
- Oracle 需要 `separate` 阶段 MossFormer2 输出。

## 10. 复现最短路径

```bash
conda activate cocktail
cd <project_root>
python run.py train_wesep --work-root ./data
python run.py eval --work-root ./data
```

## 11. 不要做的事

- 不要改 `预训练spex结果修正_*.plan.md` 等历史 plan 文件（用户明确要求）。
- 不要把 ECAPA+MossFormer2 加回最终对比表（已精简）。
- 不要用 ClearerVoice 8 kHz SpEx+ 预训练替代 WeSep（已证伪 -7.9 dB）。
- 不要仅根据 AttentionTSE 的 delta 写正面结论。

## 12. 文档地图

- 输出清单 → [outputs_index.md](outputs_index.md)
- 正式报告 → [experiment_report.md](experiment_report.md)
- 数值速查 → [experiment_log.md](experiment_log.md)
- 权重说明 → [checkpoints.md](checkpoints.md)
