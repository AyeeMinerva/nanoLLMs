"""用同一首诗的开头比较 greedy、top-k、top-p 和 temperature。"""

import argparse
import sys
from pathlib import Path

import torch

from model import load_for_eval

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="春眠不觉晓，")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "notes" / "samples.md")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model, tokenizer = load_for_eval(ROOT / "ckpt" / "best.pt")
    model = model.to(args.device).eval()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt 不能为空")

    strategies = {
        "greedy": {"temperature": 0},
        "top-k=20": {"top_k": 20},
        "top-p=0.9": {"top_p": 0.9},
        "temperature=0.7": {"temperature": 0.7},
    }
    lines = ["# 同一 prompt 下的生成样例", "",
             f"prompt：`{args.prompt}`；每组生成 {args.tokens} 个 token；seed={args.seed}。",
             "byte-level BPE 可能生成不完整的 UTF-8 序列；仅展示时用替换字符解码。", ""]

    for name, options in strategies.items():
        torch.manual_seed(args.seed)
        ids = model.generate(prompt_ids, args.tokens, **options)[0].tolist()
        # 样本可以终止于任意字节：仅显示时容错，tokenizer.decode 本身仍保持严格无损。
        text = b"".join(tokenizer.vocab[i] for i in ids).decode("utf-8", errors="replace")
        lines.extend([f"## {name}", "", "```text", text, "```", ""])
        print(f"{name}: {text}\n", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
