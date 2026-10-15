# GPT-2

本项目是直接基于[nanoGPT: karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)所进行的复现和基于个人理解的修改。

GPT-2 的最简实现: 单卡训练 / 推理 / 加载官方权重。

| 文件 | 内容 | 文档 |
|---|---|---|
| `config.py` | 模型结构超参 `GPT2Config` | [model.md](model.md) |
| `model.py` | 模型 `GPT2` | [model.md](model.md) |
| `train_config.py` | 训练超参 `TrainConfig` | [train.md](train.md) |
| `train.py` | 单卡训练 | [train.md](train.md) |
| `sample.py` | 文本生成 | [train.md](train.md) |
| `pretrained.py` | 加载 OpenAI 官方权重 | [train.md](train.md) |

## 用法

在仓库根目录运行 (环境见 [ENVIRONMENT.md](../../ENVIRONMENT.md)):

```powershell
python models/GPT-2/train.py    # 训练, 设置在 train_config.py / train.py 底部
python models/GPT-2/sample.py   # 生成, 设置在 sample.py 的 SampleConfig
```

`sample.py` 默认 `init_from='gpt2'` 加载官方 124M 权重; 改为 `'resume'` 则加载自己训练的 `out_dir/ckpt.pt`。

## 已验证 (RTX 5070 Ti Laptop 12GB)

- 124M, `batch_size=8`, `block_size=1024`, bf16: 峰值显存 9.3 GiB, 约 3.8 万 tok/s
- 官方权重加载后生成通顺文本
- 训练 → 保存 → resume → sample 全流程
