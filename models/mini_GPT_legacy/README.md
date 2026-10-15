# mini-GPT：复现流程

本项目是直接基于[llm-beginner: nndl/llm-beginner](https://github.com/nndl/llm-beginner)的任务2所进行的复现和基于个人理解的修改。
此文档部分由人工指导AI生成故可能存在错漏或过时。

从零训练 TinyStories、选择模型规模及计算 M/B 参数量，见 [TinyStories 训练指南](notes/tinystories.md)。

本目录的代码按当前仓库结构组织，不使用原任务示例中的 `src/` 目录：

```text
models/mini_GPT/
├── data/
│   ├── download.py       # 准备唐诗 train/dev
│   ├── train.txt
│   └── dev.txt
├── tokenizer.py           # byte-level BPE
├── model.py               # RoPE + causal attention + KV cache + MiniGPT
├── sampling.py            # greedy / temperature / top-k / top-p
├── train.py               # 训练、验证、保存 checkpoint
├── generate.py             # 生成四组样例
├── eval/run.py             # 当前目录结构下的原任务评测逻辑
├── ckpt/                  # 运行后生成 tokenizer.json / best.pt
├── results/               # 运行结果：训练日志、评测、生成文本、曲线（已忽略）
└── notes/model.md
```

## 1. 准备唐诗

在仓库根目录执行：

```bash
cd models/mini_GPT
python data/download.py --dataset poetry
```

脚本优先读取 `data/poetryFromTang.txt`；文件不存在时会从原任务仓库下载。随后生成：

- `data/train.txt`
- `data/dev.txt`
- `data/dataset_info.json`

当前唐诗 quick-start 的评测阈值是困惑度 `< 50`。训练后运行 `python plot_training.py`，曲线会写入 `results/train_curve.png`。

## 2. 检查 CUDA

先检查当前 Python 环境是否安装了 CUDA 版 PyTorch：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

训练脚本默认选择：有 CUDA 就用 `cuda`，否则用 `cpu`。也可以显式指定：

```bash
python train.py --device cuda
```

如果 `torch.cuda.is_available()` 是 `False`，即使命令带了 `--device cuda` 也不能使用 GPU；需要在当前虚拟环境重新安装与驱动匹配的 CUDA 版 PyTorch。安装命令以 PyTorch 官网为准，不要只根据显卡型号猜 wheel。

## 3. 训练

CPU 可复现的 quick-start 配置如下；训练脚本会训练 BPE、编码数据、训练模型，并把验证集最优模型保存到 `ckpt/best.pt`：

```bash
python train.py \
  --steps 1500 \
  --eval-interval 100 \
  --eval-steps 10 \
  --batch-size 32 \
  --block-size 128 \
  --vocab-size 320 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 128 \
  --dropout 0.15 \
  --lr 0.0005 \
  --device cpu
```

GPU 可直接把最后一项换成：

```bash
python train.py --steps 1500 --eval-interval 100 --eval-steps 10 --batch-size 32 --block-size 128 --vocab-size 320 --n-layer 4 --n-head 4 --n-embd 128 --dropout 0.15 --lr 0.0005 --device cuda
```

其他模型超参数也可以按显存增大，但扩大模型或词表会影响困惑度阈值。唐诗很小，训练步数过多会过拟合；`train.py` 按 dev loss 保存最佳 checkpoint，并在 dev 长时间不改善时早停。

训练产物：

- `ckpt/tokenizer.json`
- `ckpt/best.pt`
- `results/train_log.json`

若要完全重新训练，删除 `ckpt/` 后重跑：

```bash
rm -rf ckpt
```

Windows PowerShell：

```powershell
Remove-Item -Recurse -Force ckpt
```

## 4. 运行原任务评测

评测逻辑仍然是原任务的三个检测：tokenizer 往返、KV cache 等价性、dev 困惑度；只把导入路径和返回值适配到了本仓库的平铺结构：

```bash
python eval/run.py
```

结果写入 `results/result.json`。一次成功的唐诗结果示例：

```json
[
  {"test": "tokenizer_roundtrip", "pass": true},
  {"test": "kv_cache_equivalence", "pass": true},
  {"test": "perplexity_on_dev", "pass": true, "perplexity": 24.28}
]
```

`kv_cache_equivalence` 要求增量和全量 logits 最大误差小于 `1e-4`。评测会自动从 `best.pt` 中恢复模型配置和 tokenizer。

## 5. 生成样例

```bash
python generate.py \
  --prompt "空，" \
  --tokens 80 \
  --seed 1337

python generate.py --prompt "空山新雨" --tokens 120 --seed 1337
```

脚本会依次展示：

- `temperature=0`：greedy，始终选最大 logit
- `top-k=20`：只在概率最高的 20 个 token 中采样
- `top-p=0.9`：按累计概率截断后采样
- `temperature=0.7`：改变分布尖锐程度后采样

结果同时写入 `results/samples.md`。生成使用 KV cache；当缓存到达 `block_size` 时，代码自动退回到最近上下文重新预填充。

## 6. 从头到尾的一组命令

```bash
cd models/mini_GPT
python data/download.py --dataset poetry
python train.py --steps 1500 --eval-interval 100 --eval-steps 10 --batch-size 32 --block-size 128 --vocab-size 320 --n-layer 4 --n-head 4 --n-embd 128 --dropout 0.15 --lr 0.0005 --device cpu
python eval/run.py
python plot_training.py
python generate.py --prompt "春眠不觉晓，" --tokens 80 --seed 1337
```

GPU 版本只需将训练命令的 `--device cpu` 改为 `--device cuda`，并先确认 CUDA 检查命令返回 `True`。
