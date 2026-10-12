"""任务二自检：tokenizer / KV cache / 困惑度。"""
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # from src.* —— 学生实现
sys.path.insert(0, str(ROOT.parent))    # from _eval_harness —— 共用运行壳

from _eval_harness import run_tests


def load_dataset_info():
    path = ROOT / "data" / "dataset_info.json"
    if not path.exists():
        return {"dataset": "unknown", "ppl_threshold": 200}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"dataset": "unknown", "ppl_threshold": 200}


def test_tokenizer_roundtrip():
    from src.tokenizer import BPETokenizer
    tok_path = ROOT / "ckpt" / "tokenizer.json"
    if not tok_path.exists():
        return {"test": "tokenizer_roundtrip", "pass": None,
                "skip": "ckpt/tokenizer.json 不存在"}
    tok = BPETokenizer.from_pretrained(str(tok_path))
    samples = ["床前明月光", "Hello, world!", "深度学习需要数学基础"]
    fails = []
    for s in samples:
        ids = tok.encode(s)
        s2 = tok.decode(ids)
        if s != s2:
            fails.append({"in": s, "out": s2})
    return {"test": "tokenizer_roundtrip", "pass": not fails, "failures": fails}


def test_kv_cache_equivalence():
    from src.model import load_for_eval
    ckpt = ROOT / "ckpt" / "best.pt"
    if not ckpt.exists():
        return {"test": "kv_cache_equivalence", "pass": None, "skip": "ckpt/best.pt 不存在"}
    model, tok = load_for_eval(str(ckpt))
    model.eval()
    ids = torch.tensor([tok.encode("从前有座山")], dtype=torch.long)
    with torch.no_grad():
        logits_full = model(ids)
        cache = None
        logits_inc = []
        for i in range(ids.size(1)):
            out, cache = model(ids[:, i:i + 1], kv_cache=cache, return_cache=True)
            logits_inc.append(out)
        logits_inc = torch.cat(logits_inc, dim=1)
    diff = (logits_full - logits_inc).abs().max().item()
    return {"test": "kv_cache_equivalence", "pass": diff < 1e-4,
            "max_abs_diff": diff}


def test_perplexity_on_dev():
    from src.model import load_for_eval
    ckpt = ROOT / "ckpt" / "best.pt"
    dev_path = ROOT / "data" / "dev.txt"
    info = load_dataset_info()
    threshold = float(info.get("ppl_threshold", 200))
    if not ckpt.exists():
        return {"test": "perplexity_on_dev", "pass": None, "skip": "ckpt/best.pt 不存在"}
    if not dev_path.exists():
        return {"test": "perplexity_on_dev", "pass": None,
                "skip": "data/dev.txt 不存在；先跑 data/download.py，或训练时留出 dev split"}
    model, tok = load_for_eval(str(ckpt))
    model.eval()

    # 按模型上下文长度分块累加 NLL，再统一求困惑度：一次性喂超长序列会超出训练长度——
    # 固定 causal mask 的实现会直接 shape 报错，RoPE 实现则会让 PPL 落进外推区而失真。
    block = getattr(model, "block_size", None) or getattr(model, "max_seq_len", 256)
    ids = tok.encode(dev_path.read_text(encoding="utf-8"))[:4096]  # 自检只估量级，限 token 数控耗时
    nll, n_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, max(1, len(ids) - 1), block):
            window = ids[i:i + block + 1]
            if len(window) < 2:
                break
            chunk = torch.tensor([window], dtype=torch.long)
            logits = model(chunk)
            nll += F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1), reduction="sum").item()
            n_tok += chunk.size(1) - 1
    if n_tok == 0:
        return {"test": "perplexity_on_dev", "pass": None,
                "skip": "dev.txt 编码后 token 数不足，无法计算困惑度"}
    ppl = math.exp(nll / n_tok)
    return {"test": "perplexity_on_dev", "pass": ppl < threshold,
            "perplexity": round(ppl, 2),
            "threshold": threshold,
            "n_tokens": n_tok,
            "dataset": info.get("dataset", "unknown")}


if __name__ == "__main__":
    run_tests([test_tokenizer_roundtrip, test_kv_cache_equivalence,
               test_perplexity_on_dev], ROOT)
