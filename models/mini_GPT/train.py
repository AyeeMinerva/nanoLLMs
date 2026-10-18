"""Train MiniGPT on TinyStories.

Run from the repository root with:
    python models/mini_GPT/train.py

Data and output paths themselves are resolved relative to config.py, not cwd.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

try:  # Support both package imports and direct script execution.
    from .config import ModelConfig, TrainConfig
    from .model import MiniGPT
    from .tokenizer import ByteLevelBPETokenizer
except ImportError:  # pragma: no cover - direct `python train.py` fallback
    from config import ModelConfig, TrainConfig
    from model import MiniGPT
    from tokenizer import ByteLevelBPETokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_metadata(data_path: Path, tokenizer_path: Path, cfg: TrainConfig) -> dict:
    stat = data_path.stat()
    return {
        "source": str(data_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "tokenizer_sha256": _sha256(tokenizer_path),
        "text_column": cfg.text_column,
        "eos_token_id": ByteLevelBPETokenizer.load(tokenizer_path).eos_token_id,
        "dtype": "uint32",
        "format_version": 1,
    }


def _build_token_cache(
    data_path: Path,
    tokenizer: ByteLevelBPETokenizer,
    tokenizer_path: Path,
    cfg: TrainConfig,
    split: str,
) -> Path:
    """Encode a parquet split once, saving token IDs as a memory-mappable array."""
    import pyarrow.parquet as parquet

    if not data_path.is_file():
        raise FileNotFoundError(f"parquet file not found: {data_path}")

    parquet_file = parquet.ParquetFile(data_path)
    if cfg.text_column not in parquet_file.schema.names:
        columns = ", ".join(parquet_file.schema.names)
        raise ValueError(
            f"column {cfg.text_column!r} not found in {data_path}; "
            f"available columns: {columns}"
        )

    cache_dir = Path(cfg.token_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token_path = cache_dir / f"{split}.bin"
    metadata_path = cache_dir / f"{split}.json"
    expected_metadata = _cache_metadata(data_path, tokenizer_path, cfg)

    if token_path.is_file() and metadata_path.is_file():
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_metadata = None
        if cached_metadata == expected_metadata:
            print(f"using {split} token cache: {token_path}")
            return token_path

    print(f"encoding {split} split into token cache ...", flush=True)
    temporary_path = token_path.with_suffix(".bin.tmp")
    total_tokens = 0
    try:
        with temporary_path.open("wb") as output:
            for batch in parquet_file.iter_batches(
                columns=[cfg.text_column],
                batch_size=cfg.parquet_batch_size,
            ):
                texts = [
                    str(text)
                    for text in batch.column(0).to_pylist()
                    if text is not None
                ]
                if not texts:
                    continue
                encoded = tokenizer.encode_batch(texts, add_eos=True)
                for token_ids in encoded:
                    if token_ids:
                        ids = np.asarray(token_ids, dtype=np.uint32)
                        ids.tofile(output)
                        total_tokens += ids.size

        if total_tokens <= cfg.block_size:
            raise ValueError(
                f"{split} split has {total_tokens} tokens; needs more than "
                f"block_size={cfg.block_size} to create next-token batches"
            )
        temporary_path.replace(token_path)
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2),
            encoding="utf-8",
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"cached {total_tokens:,} {split} tokens at {token_path}", flush=True)
    return token_path


def _load_token_data(path: Path) -> np.memmap:
    data = np.memmap(path, dtype=np.uint32, mode="r")
    return data


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous next-token training windows from a token array."""
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError(
            f"token sequence length ({len(data)}) must exceed block_size ({block_size})"
        )

    starts = np.random.randint(0, max_start, size=batch_size)
    x_np = np.stack([data[start : start + block_size] for start in starts])
    y_np = np.stack([data[start + 1 : start + block_size + 1] for start in starts])
    x = torch.from_numpy(x_np.astype(np.int64, copy=False))
    y = torch.from_numpy(y_np.astype(np.int64, copy=False))

    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def configure_optimizer(model: MiniGPT, cfg: TrainConfig, device: torch.device):
    """Use AdamW with weight decay on matrix parameters only."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    decay = [parameter for parameter in parameters if parameter.ndim >= 2]
    no_decay = [parameter for parameter in parameters if parameter.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        fused=(device.type == "cuda"),
    )


def get_learning_rate(step: int, cfg: TrainConfig) -> float:
    if not cfg.decay_lr:
        return cfg.learning_rate
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_iters)
    if step >= cfg.lr_decay_iters:
        return cfg.min_lr

    progress = (step - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + cosine * (cfg.learning_rate - cfg.min_lr)


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: np.ndarray,
    validation_data: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    autocast_dtype: str,
) -> dict[str, float]:
    model.eval()
    losses = {}
    context = autocast_context(device, autocast_dtype)
    for split, data in (("train", train_data), ("validation", validation_data)):
        batch_losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, cfg.block_size, device)
            with context:
                _, loss = model(x, targets=y)
            batch_losses.append(loss.float().item())
        losses[split] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


def _model_config(cfg: TrainConfig, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
        rope_base=cfg.rope_base,
    )


def _save_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    cfg: TrainConfig,
    step: int,
    best_val_loss: float,
    tokenizer_path: Path,
) -> None:
    serialized_train_config = asdict(cfg)
    for key, value in serialized_train_config.items():
        if isinstance(value, Path):
            serialized_train_config[key] = str(value)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": serialized_train_config,
            "step": step,
            "best_val_loss": best_val_loss,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": _sha256(tokenizer_path),
        },
        path,
    )


def train(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    if cfg.dtype == "float16" and device.type != "cuda":
        raise ValueError("float16 autocast is configured for CUDA; use float32 or bfloat16 on CPU")
    if cfg.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("the selected CUDA device does not support bfloat16")

    tokenizer_path = Path(cfg.tokenizer_path)
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_path)
    train_cache = _build_token_cache(
        Path(cfg.train_data_path), tokenizer, tokenizer_path, cfg, "train"
    )
    validation_cache = _build_token_cache(
        Path(cfg.validation_data_path), tokenizer, tokenizer_path, cfg, "validation"
    )
    train_data = _load_token_data(train_cache)
    validation_data = _load_token_data(validation_cache)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    model_config = _model_config(cfg, tokenizer.vocab_size)
    model = MiniGPT(model_config).to(device)
    optimizer = configure_optimizer(model, cfg, device)
    print(f"device={device}, dtype={cfg.dtype}", flush=True)
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"vocab_size={tokenizer.vocab_size}, eos_token_id={tokenizer.eos_token_id}", flush=True)

    latest_path = cfg.out_dir / cfg.latest_checkpoint_name
    best_path = cfg.out_dir / cfg.checkpoint_name
    start_step = 0
    best_val_loss = math.inf
    if cfg.resume_from is not None:
        resume_path = Path(cfg.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("model_config") != asdict(model_config):
            raise ValueError("checkpoint model configuration differs from current TrainConfig")
        if checkpoint.get("tokenizer_sha256") != _sha256(tokenizer_path):
            raise ValueError("checkpoint was trained with a different tokenizer")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"resumed from {resume_path} at step {start_step}", flush=True)

    log_path = cfg.results_dir / cfg.train_log_name
    if cfg.resume_from is not None and log_path.is_file():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log = []
    else:
        log = []

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and cfg.dtype == "float16"),
    )
    should_stop = False
    stale_evaluations = 0
    elapsed_start = time.perf_counter()

    for step in range(start_step, cfg.max_iters + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            metrics = estimate_loss(
                model,
                train_data,
                validation_data,
                cfg,
                device,
                cfg.dtype,
            )
            improved = metrics["validation"] < best_val_loss
            if improved:
                best_val_loss = metrics["validation"]
                stale_evaluations = 0
            elif step > start_step:
                stale_evaluations += 1

            record = {
                "step": step,
                "train_loss": metrics["train"],
                "validation_loss": metrics["validation"],
                "learning_rate": get_learning_rate(min(step, cfg.max_iters - 1), cfg),
                "elapsed_seconds": time.perf_counter() - elapsed_start,
            }
            log.append(record)
            log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
            print(
                f"step {step}: train loss {metrics['train']:.4f}, "
                f"validation loss {metrics['validation']:.4f}",
                flush=True,
            )

            _save_checkpoint(
                latest_path,
                model,
                optimizer,
                model_config,
                cfg,
                step,
                best_val_loss,
                tokenizer_path,
            )
            if improved:
                _save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    model_config,
                    cfg,
                    step,
                    best_val_loss,
                    tokenizer_path,
                )
                print(f"saved best checkpoint: {best_path}", flush=True)

            if (
                cfg.early_stopping_patience is not None
                and stale_evaluations >= cfg.early_stopping_patience
            ):
                print("early stopping: validation loss stopped improving", flush=True)
                should_stop = True
            model.train()

        if step == cfg.max_iters or should_stop:
            break

        learning_rate = get_learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)

        for _ in range(cfg.gradient_accumulation_steps):
            x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, device)
            with autocast_context(device, cfg.dtype):
                _, loss = model(x, targets=y)
                scaled_loss = loss / cfg.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

        scaler.unscale_(optimizer)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.log_interval == 0:
            print(
                f"iter {step}: loss {loss.item():.4f}, lr {learning_rate:.2e}",
                flush=True,
            )

    print(f"training finished; best validation loss={best_val_loss:.4f}", flush=True)
    print(f"best checkpoint: {best_path}", flush=True)
    print(f"latest checkpoint: {latest_path}", flush=True)
    print(f"training log: {log_path}", flush=True)


def main() -> None:
    train(TrainConfig())


if __name__ == "__main__":
    main()
