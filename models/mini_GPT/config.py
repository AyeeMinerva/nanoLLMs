from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    rope_base: float = 10000.0