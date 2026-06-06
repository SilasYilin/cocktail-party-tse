# 数据目录

`work_root` 默认为仓库下的 `data/`（可在 `configs/experiment_config.json` 中修改）。

## 目录布局

```text
data/
  raw/                          # 原始数据（不入 Git）
    english/librispeech/LibriSpeech/test-clean/
    chinese/thchs30/data_thchs30/train/
    diy/recordings/spk01/*.wav
    noise/thchs_resource/       # 或 noise/musan/noise/
  processed/mixtures/           # 混合样本与 metadata.csv（运行 mix 后生成）
  downloads/                    # OpenSLR 等 tar 包缓存（可选）
```

## 自动发现

`auto_discover_data=true` 时，流水线会在 `raw/` 下查找 english、chinese、diy 目录；至少需要一个可用语料源。

若数据放在其他路径，可在配置中设置绝对路径，例如：

```json
"work_root": "/path/to/your/cocktail-data"
```

或使用命令行：

```bash
python run.py all --work-root /path/to/your/cocktail-data
```
