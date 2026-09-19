from dataclasses import dataclass

@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    def __post_init__(self):
        """实例创建完成时自动触发校验"""
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) 必须能被 n_head ({self.n_head}) 整除！"
        )
        assert self.vocab_size > 0, "vocab_size 必须为正数"
        assert self.block_size > 0, "block_size 必须为正数"