# 实验记录（速查）

> 正式报告：[experiment_report.md](experiment_report.md)  
> AI 记忆：[ai_context.md](ai_context.md)  
> 产物索引：[outputs_index.md](outputs_index.md)

## 环境

| 项 | 值 |
|----|-----|
| project_root | 仓库根目录（权重与 outputs） |
| work_root | `./data` 或外部数据目录 |
| GPU | cuda:0（实验机为 RTX 3090，物理卡 7） |
| 测试集 | 1792 条 |

## 5 路对比结果（整体）

| 模型 | ΔSI-SDR | SI-SDR_out | 2spk Δ |
|------|---------|------------|--------|
| **WeSep-BSRNN(微调)** | **+11.630** | +5.451 | **+13.584** |
| WeSep-BSRNN(预训练) | +5.320 | -0.860 | +8.527 |
| Oracle+MossFormer2 | +6.542 | +0.363 | +10.497 |
| SpEx+ 自训练 | +0.830 | -5.349 | +0.920 |
| AttentionTSE | +6.164* | -0.015 | +4.026* |

\* 塌缩假象

## WeSep 微调 2spk 分 SNR

| SNR | ΔSI-SDR |
|-----|---------|
| clean | 12.954 |
| 5 dB | 12.299 |
| 0 dB | 12.800 |
| -5 dB | 16.284 |

## 关键路径

```
checkpoints/wesep_bsrnn/avg_model.pt      # 预训练
checkpoints/wesep_bsrnn_ft/best.pt        # 微调 epoch 4
outputs/metrics_all_summary.csv           # 5 路汇总
outputs/metrics_tse_pretrained_tse_ft_*   # 微调明细
outputs/target_extracted/pretrained_tse_ft/  # 1792 wav
outputs/run_train_wesep.log                 # 训练日志
```

## 命令

```bash
python run.py train_wesep --work-root ./data
python run.py eval --work-root ./data
```

## 已移除

ECAPA+MossFormer2 评测、ClearerVoice SpEx+ 预训练（-7.9 dB 域外失败）
