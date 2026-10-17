"""TinyStories 用的 byte-level BPE tokenizer。

本文件既提供简洁的 tokenizer 封装，也可以直接运行来训练 tokenizer：

    python models/mini_GPT/tokenizer.py

训练产物的路径、词表大小和数据列名统一由 ``config.py`` 中的
``TokenizerConfig`` 管理。模型训练阶段只需要调用 ``load``，不会自动
重新训练 tokenizer。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

try:  # 支持作为包导入，也支持直接运行本文件。
    from .config import TokenizerConfig
except ImportError:  # pragma: no cover - 仅用于 python tokenizer.py
    from config import TokenizerConfig


class ByteLevelBPETokenizer:
    """对 Hugging Face ``tokenizers`` 的 byte-level BPE 薄封装。"""

    def __init__(self, tokenizer: Tokenizer, eos_token: str):
        self._tokenizer = tokenizer
        self._eos_token = eos_token

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        special_tokens: tuple[str, ...] = (),
        eos_token: str | None = None,
        length: int | None = None,
    ) -> "ByteLevelBPETokenizer":
        """从文本迭代器训练一个 byte-level BPE tokenizer。"""
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")

        pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            initial_alphabet=pre_tokenizer.alphabet(),
            special_tokens=list(special_tokens),
            show_progress=True,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer, length=length)
        if eos_token is None:
            eos_token = special_tokens[0] if special_tokens else TokenizerConfig().eos_token
        return cls(tokenizer, eos_token=eos_token)

    @classmethod
    def load(
        cls,
        path: str | Path,
        eos_token: str | None = None,
    ) -> "ByteLevelBPETokenizer":
        """从 ``tokenizer.json`` 加载 tokenizer。"""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"tokenizer not found: {path}; "
                "run tokenizer.py first to train it"
            )
        if eos_token is None:
            eos_token = TokenizerConfig().eos_token
        return cls(Tokenizer.from_file(str(path)), eos_token=eos_token)

    def save(self, path: str | Path) -> None:
        """保存为 Hugging Face 的单文件 ``tokenizer.json``。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """将一段文本编码为 token IDs，可选地在末尾追加 EOS。"""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        ids = self._tokenizer.encode(text).ids
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """将 token IDs 解码为文本。"""
        return self._tokenizer.decode(
            [int(token_id) for token_id in ids],
            skip_special_tokens=skip_special_tokens,
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        add_eos: bool = False,
    ) -> list[list[int]]:
        """批量编码文本；需要 EOS 时逐条追加，保持和 ``encode`` 一致。"""
        encodings = self._tokenizer.encode_batch(list(texts))
        batches = [encoding.ids for encoding in encodings]
        if add_eos:
            eos_token_id = self.eos_token_id
            for ids in batches:
                ids.append(eos_token_id)
        return batches

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def eos_token_id(self) -> int:
        """返回配置中的 EOS ID；未配置 EOS 时明确报错。"""
        token_id = self._tokenizer.token_to_id(self._eos_token)
        if token_id is None:
            raise RuntimeError(
                f"special token {self._eos_token!r} is not present"
            )
        return token_id


def iter_parquet_texts(
    path: str | Path,
    text_column: str,
    batch_size: int = 1024,
) -> Iterator[str]:
    """逐批读取 parquet 的文本列，避免一次性把数据集读入内存。"""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(path)
    if text_column not in parquet_file.schema.names:
        columns = ", ".join(parquet_file.schema.names)
        raise ValueError(
            f"column {text_column!r} not found in {path}; available columns: {columns}"
        )

    for record_batch in parquet_file.iter_batches(
        columns=[text_column],
        batch_size=batch_size,
    ):
        for text in record_batch.column(0).to_pylist():
            if text is not None:
                yield str(text)


def train_from_parquet(
    config: TokenizerConfig | None = None,
) -> ByteLevelBPETokenizer:
    """使用训练集 parquet 的文本列训练并保存 tokenizer。"""
    config = config or TokenizerConfig()
    data_path = Path(config.train_data_path)
    output_path = Path(config.output_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"training parquet not found: {data_path}")

    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(data_path)
    if config.text_column not in parquet_file.schema.names:
        columns = ", ".join(parquet_file.schema.names)
        raise ValueError(
            f"column {config.text_column!r} not found in {data_path}; "
            f"available columns: {columns}"
        )

    tokenizer = ByteLevelBPETokenizer.train(
        texts=iter_parquet_texts(
            data_path,
            text_column=config.text_column,
            batch_size=config.parquet_batch_size,
        ),
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=(config.eos_token,),
        eos_token=config.eos_token,
        length=parquet_file.metadata.num_rows,
    )
    tokenizer.save(output_path)
    return tokenizer


def main() -> None:
    """使用 config.py 中的默认设置训练 TinyStories tokenizer。"""
    config = TokenizerConfig()
    tokenizer = train_from_parquet(config)
    print(f"saved tokenizer to {config.output_path}")
    print(f"vocab_size={tokenizer.vocab_size}, eos_token_id={tokenizer.eos_token_id}")


if __name__ == "__main__":
    main()
