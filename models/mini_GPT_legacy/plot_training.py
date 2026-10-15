"""把 results/train_log.json 画成训练曲线：python plot_training.py。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "train_log.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "train_curve.png")
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("训练日志为空")
    rows.sort(key=lambda row: row["step"])
    steps = [row["step"] for row in rows]
    best = min(rows, key=lambda row: row["dev_loss"])

    fig, (loss_ax, ppl_ax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    loss_ax.plot(steps, [row["train_loss"] for row in rows], "o-", label="train loss")
    loss_ax.plot(steps, [row["dev_loss"] for row in rows], "o-", label="dev loss")
    loss_ax.set_ylabel("cross-entropy loss")
    loss_ax.legend()

    ppl_ax.plot(steps, [row["dev_ppl"] for row in rows], "o-", color="tab:green", label="dev perplexity")
    ppl_ax.set_xlabel("training step")
    ppl_ax.set_ylabel("perplexity")
    ppl_ax.legend()

    for ax in (loss_ax, ppl_ax):
        ax.axvline(best["step"], color="gray", ls="--", alpha=0.8,
                   label=f"best step: {best['step']}")
        ax.grid(alpha=0.25)
    loss_ax.legend()
    ppl_ax.legend()
    fig.suptitle("mini-GPT training")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)  # 后缀 .png / .pdf / .svg 决定格式
    plt.close(fig)
    print(f"saved: {args.output}")
    print(f"best dev: step={best['step']}, loss={best['dev_loss']:.3f}, ppl={best['dev_ppl']:.2f}")


if __name__ == "__main__":
    main()
