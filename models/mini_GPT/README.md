# mini_GPT

在 TinyStories 上训练一个小型 decoder-only GPT。模型、数据预处理、tokenizer 和训练产物各自分工：

- `model.py`：MiniGPT、RoPE、KV cache 和生成接口；
- `config.py`：数据路径、tokenizer 设置、模型结构及训练参数；
- `tokenizer.py`：Hugging Face `tokenizers` 的 byte-level BPE 薄封装和 tokenizer 训练入口；
- `train.py`：编码/缓存数据、训练模型、验证和保存 checkpoint；
- `plot.py`：读取训练日志并绘制 train/validation loss；
- `chat.py`：加载 checkpoint，进行单次文本续写或交互式生成。

数据和产物路径都由 `config.py` 基于模块位置定义，不依赖当前工作目录。下面的命令示例从仓库根目录执行。

## 依赖

本实现使用以下新增依赖：

- `tokenizers`：Rust 实现的 BPE tokenizer；
- `pyarrow`：分批读取 TinyStories Parquet。

它们已加入仓库的 `pyproject.toml`、`requirements.txt` 和 `uv.lock`。在仓库根目录安装/同步依赖：

```bash
uv sync --locked
```

如果使用 pip 管理环境：

```bash
pip install -r requirements.txt
```

不需要额外安装 `pandas`、`datasets` 或 `sentencepiece`。tokenizer 训练和 Parquet 预处理主要使用 CPU；GPU 用于后续模型训练。

## 数据和 tokenizer

默认读取：

```text
data/tinystories/tinystories-train.parquet
data/tinystories/tinystories-validation.parquet
```

两个 Parquet 文件都需要包含 `text` 列。TinyStories 训练集已有 tokenizer 训练入口：

```bash
python models/mini_GPT/tokenizer.py
```

默认使用 byte-level BPE，目标词表大小为 8192，EOS token 为 `<|endoftext|>`。可在 `config.py` 的 `TokenizerConfig` 中调整这些设置。tokenizer 只从训练 split 学习；验证 split 不参与 tokenizer 训练。

训练好的单文件 tokenizer 保存到：

```text
models/mini_GPT/ckpt/tokenizer.json
```

模型训练不会自动重训 tokenizer。若该文件不存在，先运行 tokenizer 训练命令。

## 模型训练

```bash
python models/mini_GPT/train.py
```

训练参数集中在 `config.py` 的 `TrainConfig` 中修改，例如 batch size、累积步数、上下文长度、模型层数/宽度、精度、学习率、训练步数和评估间隔。默认设备为自动选择 CUDA 或 CPU，默认精度为 `float32`。支持的混合精度为 `float16` 和 `bfloat16`；在支持 BF16 的 CUDA GPU（例如 RTX 5070 Ti）上，可将 `dtype` 调整为 `"bfloat16"`。显存不足时，可先减小 `batch_size` 或 `block_size`，也可以用 `gradient_accumulation_steps` 增大有效 batch size。

首次运行会分别将训练集和验证集编码为模型专属的 uint32 token cache：

```text
models/mini_GPT/cache/tinystories/train.bin
models/mini_GPT/cache/tinystories/validation.bin
```

缓存元数据记录数据文件信息、tokenizer 哈希、文本列和 EOS ID。相关信息变化时，对应缓存会自动重建。token cache 是可删除的派生产物；重新生成不会改动原始 Parquet 或 tokenizer。

训练输出：

```text
models/mini_GPT/ckpt/best.pt       # 验证 loss 最佳 checkpoint
models/mini_GPT/ckpt/latest.pt     # 最近一次评估时的 checkpoint，供续训使用
models/mini_GPT/results/train_log.json
```

`best.pt` 保存模型配置、权重、优化器状态、训练步数及 tokenizer 哈希。不要将 checkpoint 与另一份 tokenizer 搭配使用。

### 续训

编辑 `config.py` 中的 `TrainConfig.resume_from`，指向已有 checkpoint，例如：

```python
resume_from: Path | None = Path(__file__).resolve().parent / "ckpt" / "latest.pt"
```

再运行同一训练命令。恢复时会校验模型结构和 tokenizer 哈希。完成续训后，将该值改回 `None`，即可从头开始下一次实验。

## 绘制 loss 曲线

训练产生 `train_log.json` 后运行：

```bash
python models/mini_GPT/plot.py
```

图像保存为：

```text
models/mini_GPT/results/loss_curve.png
```

## 推理与交互聊天

推荐使用 `chat.py`，它会从 `config.py` 中读取默认 checkpoint、tokenizer 和采样设置。

### 单次文本续写

```bash
python models/mini_GPT/chat.py --prompt "Once upon a time"
```

### 交互式生成

不提供 `--prompt` 时进入交互模式：

```bash
python models/mini_GPT/chat.py
```

输入 `/exit` 或 `/quit` 退出。也可以临时覆盖采样参数：

```bash
python models/mini_GPT/chat.py \
    --max-new-tokens 160 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9
```

可用参数包括：

- `--checkpoint`：覆盖默认 checkpoint 路径；
- `--tokenizer`：覆盖默认 `tokenizer.json` 路径；
- `--device`：使用 `auto`、`cpu` 或具体 PyTorch 设备；
- `--max-new-tokens`：最多生成的 token 数；
- `--temperature`：温度为 `0` 时使用 greedy decoding；
- `--top-k`：限制采样候选 token 数，`0` 表示不启用；
- `--top-p`：nucleus sampling 阈值。

当前 MiniGPT 是在 TinyStories 上进行 next-token prediction 的基础语言模型，并没有经过 instruction tuning 或对话数据微调。因此交互模式只是把多轮文本拼接成 prompt，不能保证像 ChatGPT 一样遵循指令；它更适合观察模型的文本续写能力。

推理脚本会校验 checkpoint 中记录的 tokenizer SHA-256 和词表大小，避免误用不匹配的 tokenizer。

### Python API 示例

从仓库根目录运行以下 Python 代码，加载验证集最佳 checkpoint 并生成文本：

```python
import torch

from models.mini_GPT.config import ModelConfig, TrainConfig
from models.mini_GPT.model import MiniGPT
from models.mini_GPT.tokenizer import ByteLevelBPETokenizer

cfg = TrainConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = cfg.out_dir / cfg.checkpoint_name
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

model_config = ModelConfig(**checkpoint["model_config"])
model = MiniGPT(model_config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

tokenizer = ByteLevelBPETokenizer.load(cfg.tokenizer_path)
prompt = "Once upon a time"
prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)

with torch.inference_mode():
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=120,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
    )

print(tokenizer.decode(generated_ids[0].tolist()))
```

推理时调用 `model.eval()` 关闭 dropout；`generate()` 的 `max_new_tokens` 控制最多新生成的 token 数。