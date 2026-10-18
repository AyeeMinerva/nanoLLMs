from dataclasses import dataclass
from pathlib import Path


MINI_GPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MINI_GPT_DIR.parent.parent


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


@dataclass
class TokenizerConfig:
    train_data_path: Path = PROJECT_ROOT / "data" / "tinystories" / "tinystories-train.parquet"
    output_path: Path = MINI_GPT_DIR / "ckpt" / "tokenizer.json"
    text_column: str = "text"
    vocab_size: int = 8192
    min_frequency: int = 2
    parquet_batch_size: int = 1024
    eos_token: str = "<|endoftext|>"

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if self.parquet_batch_size <= 0:
            raise ValueError("parquet_batch_size must be positive")
        if not self.text_column:
            raise ValueError("text_column must not be empty")
        if not self.eos_token:
            raise ValueError("eos_token must not be empty")


@dataclass
class TrainConfig:
    # Dataset, tokenizer, cache, and output paths.
    train_data_path: Path = PROJECT_ROOT / "data" / "tinystories" / "tinystories-train.parquet"
    validation_data_path: Path = PROJECT_ROOT / "data" / "tinystories" / "tinystories-validation.parquet"
    text_column: str = "text"
    tokenizer_path: Path = TokenizerConfig.output_path
    token_cache_dir: Path = MINI_GPT_DIR / "cache" / "tinystories"
    parquet_batch_size: int = 1024
    out_dir: Path = MINI_GPT_DIR / "ckpt"
    results_dir: Path = MINI_GPT_DIR / "results"
    checkpoint_name: str = "best.pt"
    latest_checkpoint_name: str = "latest.pt"
    train_log_name: str = "train_log.json"
    loss_curve_name: str = "loss_curve.png"
    resume_from: Path | None = None

    # Model size.
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    rope_base: float = 10000.0

    # Optimization and batching.
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_iters: int = 10_000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int | None = None

    # Evaluation and runtime.
    eval_interval: int = 250
    eval_iters: int = 20
    log_interval: int = 10
    early_stopping_patience: int | None = 8
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 1337

    def __post_init__(self):
        if self.lr_decay_iters is None:
            self.lr_decay_iters = self.max_iters

        positive_ints = {
            "parquet_batch_size": self.parquet_batch_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_iters": self.max_iters,
            "eval_interval": self.eval_interval,
            "eval_iters": self.eval_iters,
            "log_interval": self.log_interval,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if not self.text_column:
            raise ValueError("text_column must not be empty")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("the attention head size must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        if self.lr_decay_iters <= 0:
            raise ValueError("lr_decay_iters must be positive")
        if self.decay_lr and self.lr_decay_iters <= self.warmup_iters:
            raise ValueError("lr_decay_iters must exceed warmup_iters when decay_lr is enabled")
        if self.learning_rate <= 0 or self.min_lr < 0:
            raise ValueError("learning_rate must be positive and min_lr non-negative")
        if self.min_lr > self.learning_rate:
            raise ValueError("min_lr must not exceed learning_rate")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in the range [0, 1)")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be 'float32', 'float16', or 'bfloat16'")


@dataclass
class InferenceConfig:
    checkpoint_path: Path = MINI_GPT_DIR / "ckpt" / "best.pt"
    tokenizer_path: Path = MINI_GPT_DIR / "ckpt" / "tokenizer.json"
    device: str = "auto"
    max_new_tokens: int = 120
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.8
    system_prompt: str = "You are a helpful assistant. Continue the conversation naturally."

    def __post_init__(self):
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the range (0, 1]")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")