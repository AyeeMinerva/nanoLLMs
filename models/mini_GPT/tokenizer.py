"""BBPE 分词器"""

import json
from pathlib import Path


class BPETokenizer:
    """从 256 个原始字节出发，按频率学习相邻 token 的合并规则。"""

    def __init__(self, vocab_size: int = 512):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        self.max_vocab_size = vocab_size
        self.vocab = [bytes([i]) for i in range(256)]  # 下标即 token ID
        self.merges = {}  # (左 id, 右 id) -> 新 id；新 id 即合并顺序

    @property
    def vocab_size(self) -> int:
        """当前实际词表大小，供模型的 embedding 使用。"""
        return len(self.vocab)

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """从左到右合并一对相邻 token（重叠时只合并一次）。"""
        result = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def train(self, text: str) -> None:
        """每轮选择当前语料中出现次数最多的相邻对，重新学习词表。"""
        self.vocab = [bytes([i]) for i in range(256)]
        self.merges = {}
        ids = list(text.encode("utf-8"))

        while len(self.vocab) < self.max_vocab_size and len(ids) > 1:
            counts = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1

            pair = max(counts, key=counts.get)  # 同频时选最先出现的对
            if counts[pair] < 2:
                break  # 只出现一次的片段不值得加入词表
            new_id = len(self.vocab)
            self.merges[pair] = new_id
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
            ids = self._merge(ids, pair, new_id)

    def encode(self, text: str) -> list[int]:
        """从 UTF-8 字节出发，按训练时的先后顺序应用合并。"""
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            pair = min(
                (p for p in zip(ids, ids[1:]) if p in self.merges),
                key=self.merges.__getitem__,
                default=None,
            )
            if pair is None:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """先拼接所有 token 的字节，再统一按 UTF-8 解码。"""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8")

    def save_pretrained(self, path: str | Path) -> None:
        """将词表（十六进制字节）和有序合并规则写入 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_vocab_size": self.max_vocab_size,
            "vocab": [token.hex() for token in self.vocab],
            "merges": [[*pair, new_id] for pair, new_id in self.merges.items()],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BPETokenizer":
        """从 save_pretrained 写出的 JSON 文件恢复词表与合并顺序。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(vocab_size=data.get("max_vocab_size", len(data["vocab"])))
        tokenizer.vocab = [bytes.fromhex(token) for token in data["vocab"]]
        tokenizer.merges = {
            (left, right): new_id for left, right, new_id in data["merges"]
        }
        return tokenizer
