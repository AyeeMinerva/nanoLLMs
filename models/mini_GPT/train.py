"""在 data/train.txt 上训练 mini-GPT。

示例：
    python train.py --steps 3000 --vocab-size 512
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch

from model import MiniGPT, ModelConfig
from tokenizer import BPETokenizer

ROOT = Path(__file__).parent


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8, help="dev 连续几次不改善后早停")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_tokens(path, tokenizer):
    return torch.tensor(tokenizer.encode(path.read_text(encoding="utf-8")), dtype=torch.long)


def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, train_data, dev_data, args):
    model.eval()
    losses = []
    for _ in range(args.eval_steps):
        x, y = get_batch(train_data, args.batch_size, args.block_size, args.device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    train_loss = sum(losses) / len(losses)

    # 与 eval/run.py 相同：固定、非重叠的 dev 窗口；只用 dev 做选择，不反传。
    nll, n_tokens = 0.0, 0
    for i in range(0, min(len(dev_data), 4096) - 1, args.block_size - 1):
        window = dev_data[i:i + args.block_size].to(args.device)
        if window.numel() < 2:
            break
        logits = model(window[None, :])
        nll += torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            window[1:].reshape(-1), reduction="sum"
        ).item()
        n_tokens += window.numel() - 1
    model.train()
    return {"train_loss": train_loss, "dev_loss": nll / n_tokens,
            "dev_ppl": math.exp(nll / n_tokens)}


def main():
    args = read_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data_dir = ROOT / "data"
    ckpt_dir = ROOT / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    train_path = data_dir / "train.txt"
    dev_path = data_dir / "dev.txt"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("先运行: python data/download.py --dataset poetry")

    tokenizer_path = ckpt_dir / "tokenizer.json"
    print("训练 tokenizer ...", flush=True)
    tokenizer = BPETokenizer(args.vocab_size)
    tokenizer.train(train_path.read_text(encoding="utf-8"))
    tokenizer.save_pretrained(tokenizer_path)
    print(f"vocab_size={tokenizer.vocab_size}", flush=True)

    train_data = load_tokens(train_path, tokenizer)
    dev_data = load_tokens(dev_path, tokenizer)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_dev = math.inf
    stale = 0
    log = []

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(model, train_data, dev_data, args)
            metrics["step"] = step
            log.append(metrics)
            print(f"step {step}: train {metrics['train_loss']:.3f}, "
                  f"dev {metrics['dev_loss']:.3f}, ppl {metrics['dev_ppl']:.2f}", flush=True)
            if metrics["dev_loss"] < best_dev:
                best_dev = metrics["dev_loss"]
                stale = 0
                torch.save({
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "tokenizer": str(tokenizer_path.name),
                }, ckpt_dir / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"dev 未改善 {stale} 次，提前停止于 step {step}")
                    break

        if step == args.steps:
            break
        # 从初始 lr 平滑下降至 10%，避免末期过大的更新破坏验证集性能。
        lr = args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "train_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
