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
        assert self.n_layer > 0, "n_layer 必须为正数"
        assert self.n_head > 0, "n_head 必须为正数"
        assert self.n_embd > 0, "n_embd 必须为正数"
        assert 0.0 <= self.dropout <= 1.0, "dropout 必须在 [0, 1] 范围内"

@dataclass
class TrainConfig:
    # I/O
    out_dir: str = 'out/gpt2'
    init_from: str = 'scratch'              # 'scratch': 从零训练; 'resume': 从 out_dir/ckpt.pt 继续
    eval_interval: int = 500                # 每隔多少步评估一次 train/val loss
    eval_iters: int = 100                   # 每次评估取多少个 batch 求平均
    log_interval: int = 10                  # 每隔多少步打印一次训练 loss
    always_save_checkpoint: bool = False    # False: 只在 val loss 创新低时保存

    # data
    data_dir: str = 'data/shakespeare_char' # 需包含 train.bin / val.bin (uint16 token ids), 可选 meta.pkl
    batch_size: int = 8                     # micro-batch 大小, 受显存限制
    gradient_accumulation_steps: int = 8    # 有效 batch = batch_size * gradient_accumulation_steps

    # AdamW optimizer
    learning_rate: float = 6e-4             # 最大学习率
    max_iters: int = 20000                  # 总训练步数 (一步 = 一次参数更新)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0                  # 梯度裁剪阈值, 0.0 表示不裁剪

    # learning rate schedule: linear warmup + cosine decay
    decay_lr: bool = True
    warmup_iters: int = 1000
    lr_decay_iters: int = 20000             # 一般等于 max_iters
    min_lr: float = 6e-5                    # 一般为 learning_rate / 10

    # system
    device: str = 'cuda'
    dtype: str = 'bfloat16'                 # 'bfloat16' 或 'float32'
    compile: bool = False                   # Windows 上 torch.compile 需额外安装 triton-windows
    seed: int = 1337

    def __post_init__(self):
        """实例创建完成时自动触发校验"""
        assert self.init_from in ('scratch', 'resume'), "init_from 必须为 'scratch' 或 'resume'"
        assert self.dtype in ('bfloat16', 'float32'), "dtype 必须为 'bfloat16' 或 'float32'"
        assert self.batch_size > 0, "batch_size 必须为正数"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps 必须为正数"
        assert self.warmup_iters < self.lr_decay_iters, "warmup_iters 必须小于 lr_decay_iters"
        assert self.min_lr <= self.learning_rate, "min_lr 不能大于 learning_rate"
