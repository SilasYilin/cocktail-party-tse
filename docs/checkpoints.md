# 模型权重

权重目录为 `checkpoints/`（已加入 `.gitignore`，需自行下载）。

## ECAPA 说话人编码器（SpeechBrain，冻结用于 AttentionTSE）

目录：`checkpoints/spkrec-ecapa-voxceleb/`

需包含：`hyperparams.yaml`、`embedding_model.ckpt`、`mean_var_norm_emb.ckpt`、`classifier.ckpt`、`label_encoder.txt` 等。

```bash
pip install huggingface-hub
huggingface-cli download speechbrain/spkrec-ecapa-voxceleb \
  --local-dir checkpoints/spkrec-ecapa-voxceleb
```

## MossFormer2 分离（ClearVoice，测试集基线与 Oracle）

目录：`checkpoints/clearvoice/`

需包含 `last_best_checkpoint` 及对应的 `.pt` 权重文件。

## WeSep 预训练 BSRNN+ECAPA（VoxCeleb1，16 kHz）

目录：`checkpoints/wesep_bsrnn/`

需包含：

- `avg_model.pt`
- `config.yaml`

可从 ModelScope 或 WeSep 官方 Hub 下载 `bsrnn_ecapa_vox1` 后放到上述目录。推理入口：`cocktail_party/pretrained_tse.py`（`wesep.cli.extractor.load_model_local`）。

```bash
pip install "git+https://github.com/wenet-e2e/WeSep.git" silero-vad onnxruntime
```

## WeSep 微调（域内 SI-SDR）

目录：`checkpoints/wesep_bsrnn_ft/`

由 `python run.py train_wesep` 生成，最佳权重为 `best.pt`（从 `wesep_bsrnn/avg_model.pt` 初始化）。

## 自训练 SpEx+（对照）

目录：`checkpoints/spex_plus/best.pt`，由 `python run.py train_spex` 生成。

## AttentionTSE（训练产出）

目录：`checkpoints/attention_tse/`

由 `python run.py train` 生成，最佳权重为：

```
checkpoints/attention_tse/best.pt
```

## 文档交叉引用

- 产物路径与 CSV 字段：[outputs_index.md](outputs_index.md)
- 实验结论与数值：[experiment_report.md](experiment_report.md)
- AI 会话记忆：[ai_context.md](ai_context.md)

## 校验

```bash
test -f checkpoints/spkrec-ecapa-voxceleb/hyperparams.yaml && echo "encoder ok"
test -f checkpoints/clearvoice/last_best_checkpoint && echo "separation ok"
test -f checkpoints/wesep_bsrnn/avg_model.pt && echo "wesep pretrained ok"
test -f checkpoints/wesep_bsrnn_ft/best.pt && echo "wesep finetuned ok (after train_wesep)"
test -f checkpoints/attention_tse/best.pt && echo "attention_tse ok (after train)"
```
