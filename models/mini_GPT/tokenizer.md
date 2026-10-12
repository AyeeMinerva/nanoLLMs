# 从零理解简化版 Byte-level BPE

`tokenizer.py` 实现了任务 M1 的 `BPETokenizer`。它把文本编码为整数 ID，供模型的 embedding 使用；不依赖分词库。

## 先看接口

```python
from pathlib import Path
from tokenizer import BPETokenizer

text = Path("data/train.txt").read_text(encoding="utf-8")
tokenizer = BPETokenizer(vocab_size=512)
tokenizer.train(text)
tokenizer.save_pretrained("ckpt/tokenizer.json")

tokenizer = BPETokenizer.from_pretrained("ckpt/tokenizer.json")
ids = tokenizer.encode("春眠不觉晓")
assert tokenizer.decode(ids) == "春眠不觉晓"
print(tokenizer.vocab_size)  # 训练后实际词表大小，用于 nn.Embedding
```

路径以运行程序时的工作目录为准。如果将文件作为包导入，请按项目的导入方式调整第一行。

## 核心思路

1. **初始词表**：ID `0..255` 一一对应单个字节。训练和编码都先对文本做 UTF-8 编码，因此中文、英文及语料里没见过的字符都能表示。
2. **训练一轮**：扫描当前语料的相邻 ID 对，找出现次数最多的一对；给它分配一个新 ID，并从左到右把这一对的所有**不重叠**出现替换为新 ID。
3. **继续训练**：重复上一步，直到词表达到目标大小，或语料不足两个 token。`vocab_size` 返回实际大小，不保证一定等于构造时指定的上限。
4. **编码新文本**：从字节 ID 开始，每次寻找当前相邻对中**最早学到**的合并规则，应用这条规则；直到不能再合并。规则的新 ID 从 256 起递增，因而也记录了学习顺序。
5. **解码**：每个 ID 对应一段字节。先拼接**所有**字节，再统一执行 UTF-8 解码；不能逐 token 解码，因为一个汉字的字节可能分散在多个 token 中。

例如语料 `abab` 初始为 `[97, 98, 97, 98]`；`(97, 98)` 出现两次，第一次合并后得到 `[256, 256]`，而 `vocab[256]` 是字节 `b"ab"`。把两个 token 的字节拼起来便能还原原文。其他合并规则按频率在后续轮次学习。

`train` 重建词表和合并规则；因此训练前后、保存并加载后，`decode(encode(text)) == text` 对可编码为 UTF-8 的文本成立（前提是没有手动截断 ID 序列）。词表用十六进制字符串写入 JSON，以便无损保存任意字节；合并规则记录 `[左 ID, 右 ID, 新 ID]`，同时保存训练时的词表上限。JSON 是这个教学实现自己的格式，不兼容 Hugging Face 的 tokenizer 文件。

## 刻意保持简单的边界

- 在**整段字节流**上学习，不做按词、标点或句子的预切分，合并可能跨越字符和词的边界；实际项目通常会先预切分。
- 没有 `<eos>`、`<pad>` 等特殊 token；若训练或生成需要它们，应在设计数据流程时另行约定。
- 为易读性，每轮重新统计语料、编码时反复扫描相邻对，适合小语料学习，不适合大规模高性能训练。即使最高频对只出现一次也会合并，直至达到上限或只剩一个 token；这些低频合并对泛化帮助不大。
- `decode` 要接收完整的、能组成合法 UTF-8 的 ID 序列；任意截断后若停在汉字的字节中间，会抛出 `UnicodeDecodeError`。这不影响完整文本的往返还原。
