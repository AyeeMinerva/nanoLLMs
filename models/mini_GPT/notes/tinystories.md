# 从零训练 TinyStories：命令与模型规模

本文命令适用于当前 mini-GPT 实现。以 Windows PowerShell 为例；Python 命令也可以直接用于其他终端。所有命令都在 `models/mini_GPT` 下执行。

目标是英文短故事续写。TinyStories 原版是英文数据，应使用英文 prompt。训练出基本通顺的短故事需要足够的数据和训练预算，下面的参数是实验起点，不保证固定步数达到某个生成质量或困惑度。

## 1. 进入目录，检查环境

从仓库根目录执行：

```powershell
cd models/mini_GPT
python --version
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi
```

需要 CUDA 检查返回 `True`，才能使用 `--device cuda`。务必在实际运行训练的同一个 Python 环境中检查。

准备数据处理和绘图库：

```powershell
python -m pip install datasets pyarrow matplotlib
```

这些命令不负责安装或更换 PyTorch；CUDA 版 PyTorch 应按本机驱动和项目环境配置安装。

查看脚本支持的参数：

```powershell
python train.py --help
python generate.py --help
python plot_training.py --help
```

## 2. 先备份唐诗实验

当前训练脚本每次都会重新训练 tokenizer、随机初始化模型，并覆盖同名产物。它没有 `--resume` 或 `--output-dir`。

下面将旧实验移到忽略的 `results/` 下，不删除内容：

```powershell
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$archiveDir = "results/archive-$runStamp"
New-Item -ItemType Directory -Force $archiveDir | Out-Null
if (Test-Path ckpt) { Move-Item ckpt "$archiveDir/ckpt" }
foreach ($name in @('train_log.json', 'result.json', 'samples.md', 'train_curve.png')) {
    if (Test-Path "results/$name") { Move-Item "results/$name" "$archiveDir/$name" }
}
New-Item -ItemType Directory -Force "$archiveDir/data" | Out-Null
foreach ($name in @('train.txt', 'dev.txt', 'dataset_info.json')) {
    if (Test-Path "data/$name") { Copy-Item "data/$name" "$archiveDir/data/$name" }
}
```

如果当前 train/dev 已是完整 TinyStories，不建议再复制 GB 级文本：可以跳过上面数据备份部分，原始 parquet 仍可用于重新生成文本。

## 3. 准备 TinyStories 子集

**先从完整故事组成的小子集开始。当前 BPE 是教学实现，不适合直接对 GB 级文本学习合并规则并整段编码。**GPU 不会加速这些 Python tokenizer 循环；看到 GPU 空闲，可能只是仍在预处理。

### 已有本地 parquet：推荐直接复用

如果存在以下文件，不必重新下载全量数据：

```text
data/tinystories-train.parquet
data/tinystories-validation.parquet
```

运行以下命令，流式读取 parquet，按完整故事截取约 10 MB 训练文本、约 1 MB 验证文本。它会覆盖 `data/train.txt`、`data/dev.txt` 和元数据，不修改原始 parquet。

```powershell
@'
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path('data')
for split, filename, limit in [
    ('train', 'train.txt', 10_000_000),
    ('validation', 'dev.txt', 1_000_000),
]:
    source = pq.ParquetFile(root / f'tinystories-{split}.parquet')
    written = stories = 0
    with (root / filename).open('w', encoding='utf-8', newline='\n') as output:
        for batch in source.iter_batches(batch_size=256, columns=['text']):
            for text in batch.column(0).to_pylist():
                text = (text or '').strip()
                if not text:
                    continue
                text += '\n\n'
                output.write(text)
                written += len(text.encode('utf-8'))
                stories += 1
                if written >= limit:
                    break
            if written >= limit:
                break
    print(filename, 'stories:', stories, 'bytes:', written)

info = {'dataset': 'tinystories', 'train': 'train.txt', 'dev': 'dev.txt',
        'ppl_threshold': 10, 'note': 'Complete-story subset from official train/validation splits'}
(root / 'dataset_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
'@ | python -
```

想扩大数据时，修改其中的 `10_000_000`。先观察预处理耗时，再扩大到 50–200 MB；较大的文本和词表可能需要先改善 tokenizer 的效率。

### 没有本地数据：下载后再截取

```powershell
python data/download.py --dataset tinystories
```

这个命令会下载并导出全量数据，可能占用数 GB 磁盘。完成后运行上一节的 parquet 子集命令，再开始训练。不要直接把全量文本交给当前教学 BPE。

## 4. 选择模型配置

建议先用 3.42M 参数验证预处理速度、显存和生成流程，再考虑 11.43M。数据集变大后再增加容量，比在很小子集上反复扩大模型更合理。

| 配置 | 词表 V | 层数 L | 隐藏维 C | 注意力头 | 参数量 |
|---|---:|---:|---:|---:|---:|
| 原唐诗实验 | 320 | 4 | 128 | 4 | 0.834M = 0.000834B |
| TinyStories 小实验 | 1024 | 4 | 256 | 4 | 3.422M = 0.003422B |
| TinyStories 推荐起点 | 2048 | 6 | 384 | 6 | 11.434M = 0.011434B |
| 后续扩大实验 | 4096 | 8 | 512 | 8 | 27.317M = 0.027317B |

表中假定实际词表达到指定大小，且使用当前默认 `bias=True`。这些是参数量，不是显存要求。

### 小模型：先完成一轮实验

```powershell
python train.py --steps 10000 --eval-interval 250 --eval-steps 10 --batch-size 32 --block-size 256 --vocab-size 1024 --n-layer 4 --n-head 4 --n-embd 256 --dropout 0.1 --lr 0.0003 --weight-decay 0.1 --patience 10 --device cuda
```

### 推荐起点：约 11.43M 参数

先归档上一轮的 `ckpt/` 和 `results/` 顶层产物，再运行：

```powershell
python train.py --steps 20000 --eval-interval 500 --eval-steps 10 --batch-size 32 --block-size 256 --vocab-size 2048 --n-layer 6 --n-head 6 --n-embd 384 --dropout 0.1 --lr 0.0003 --weight-decay 0.1 --patience 10 --device cuda
```

- 显存不足时，先将 `--batch-size 32` 降到 16 或 8。当前实现没有梯度累积，降低 batch 会减少每一步处理的 token。
- 使用默认 FP32 训练，训练显存还包括梯度、AdamW 状态和中间激活，不能按权重文件大小估计。
- `--patience 10` 表示连续 10 次 dev 评估不改善后停止，可能在达到指定步数前结束。
- 日志中的 train loss 是评估时在训练集随机抽样得到的平均 loss，不是每一步的训练 loss。
- 当前 dev 评估只看前 4096 个 token，适合快速检查，不足以代表整个验证集的最终表现。
- 当前训练日志只在正常结束或早停后写入 `results/train_log.json`。中途强制终止可能没有新日志；最佳 checkpoint 会在 dev 改善时保存。
- checkpoint 不包含优化器状态、训练步数等完整恢复信息。重新执行训练命令会从头训练，不能接着旧 checkpoint 继续。

另开终端查看 GPU：

```powershell
nvidia-smi -l 2
```

按 Ctrl+C 停止监控，不影响另一个终端的训练。

## 5. 评测、画图和生成

```powershell
python eval/run.py
python plot_training.py
python generate.py --prompt "Once upon a time, a little rabbit" --tokens 120 --seed 1337 --device cuda --output results/rabbit.md
python generate.py --prompt "One day, Lily found a small box in the garden." --tokens 120 --seed 1337 --device cuda --output results/box.md
python generate.py --prompt "Tom wanted to help his friend, but" --tokens 120 --seed 1337 --device cuda --output results/friend.md
```

每次生成会比较四种采样策略。使用不同 `--output` 可以保留多组结果。

```text
ckpt/tokenizer.json       BPE 词表和合并规则
ckpt/best.pt              dev loss 最好的模型和配置
results/train_log.json    训练评估记录
results/train_curve.png   loss/PPL 曲线
results/result.json       三项自检
results/*.md              生成样例
```

`ckpt/` 和 `results/` 均已忽略。请保留成对的模型和 tokenizer：新 tokenizer 的 token ID 含义可能不同，不能只拿旧权重搭配新词表。

TinyStories 的自检阈值是 PPL < 10，但通过阈值不保证故事质量，也不能直接跨不同 tokenizer 比较 PPL。原 KV cache 测试仍使用中文短句，它检查计算等价性，不检查中文能力。

## 6. 模型的 M/B 怎么算

`M` 表示百万参数，`B` 表示十亿参数，均按十进制：

```text
1M = 1,000,000 个参数
1B = 1,000,000,000 个参数 = 1000M
模型 B 数 = 参数总数 / 1,000,000,000
```

11,433,984 个参数是约 `11.43M` 或 `0.01143B`，不是 11B。通常小于 1B 的模型直接用 M 表达更清晰。

### 查看实际训练出的模型

```powershell
python -c "from model import load_for_eval; m,_=load_for_eval('ckpt/best.pt'); n=sum(p.numel() for p in m.parameters()); size=sum(p.numel()*p.element_size() for p in m.parameters()); print(m.config); print(f'{n:,} parameters = {n/1e6:.3f}M = {n/1e9:.6f}B'); print(f'weight memory: {size/1024**2:.2f} MiB')"
```

这会加载当前 `ckpt/best.pt`，配置才是判断实际模型规模的依据。只改默认值不会改变已经保存的模型。

模型使用 embedding/output 权重绑定：`wte.weight` 与 `lm_head.weight` 是同一个参数。用 `model.parameters()` 统计会自动去重；不要直接把 `state_dict()` 所有 tensor 的元素数相加，否则会重复计数。

### 不训练也能计算计划中的参数量

以下公式适用于当前架构：MLP 扩展 4 倍、默认 bias=True、两个 LayerNorm、输出权重绑定。

```powershell
python -c "V,L,C=2048,6,384; n=V*C+L*(12*C*C+13*C)+2*C; print(f'{n:,} parameters = {n/1e6:.3f}M = {n/1e9:.6f}B')"
```

也可以直接实例化核对：

```powershell
python -c "from model import MiniGPT, ModelConfig; m=MiniGPT(ModelConfig(vocab_size=2048,n_layer=6,n_head=6,n_embd=384,block_size=256)); n=sum(p.numel() for p in m.parameters()); print(f'{n:,} parameters = {n/1e6:.3f}M = {n/1e9:.6f}B')"
```

参数量的近似主项是：

```text
N ≈ V × C + 12 × L × C²
```

| 参数 | 对规模或资源的影响 |
|---|---|
| `--n-layer` | 层数翻倍，Block 参数量约翻倍 |
| `--n-embd` | 隐藏维翻倍，Block 参数量约变成 4 倍 |
| `--vocab-size` | embedding 参数随词表线性增加；最终以实际训练出的词表大小为准 |
| `--n-head` | 固定隐藏维时基本不改变参数量，改变每个头的维度 |
| `--block-size` | RoPE 不增加可训练位置参数，但上下文变长会增加计算和显存 |
| batch、steps、lr、dropout | 不改变参数量，改变训练过程和资源消耗 |

必须满足 `n_embd % n_head == 0`，且 `n_embd // n_head` 为偶数，才能匹配当前 RoPE。

## 7. 步数与 token 预算

当前代码每个 batch 更新一次参数，无梯度累积，单 GPU 时：

```text
tokens/step = batch_size × block_size
总处理 token ≈ 已完成更新步数 × batch_size × block_size
```

推荐配置的上限预算：

```text
20000 × 32 × 256 = 163,840,000 token，约 1.64 亿
```

这包含重复看到的 token，不等于拥有 1.64 亿个不同 token。随机抽取的窗口也可能重叠。

在 tokenizer 保存后，可统计数据集的实际 token 数。此命令会再次编码文本，较大的数据上可能耗时很久：

```powershell
python -c "from pathlib import Path; from tokenizer import BPETokenizer; t=BPETokenizer.from_pretrained('ckpt/tokenizer.json'); n=len(t.encode(Path('data/train.txt').read_text(encoding='utf-8'))); print('train tokens:',n); print('equivalent passes at 20000 steps:',20000*32*256/n)"
```

等效遍历次数过多且 dev loss 上升时，应考虑增加数据、早停或减少模型容量。看到短句开始通顺后，再逐步扩大数据量和上下文；不要一次同时改变所有超参数。
