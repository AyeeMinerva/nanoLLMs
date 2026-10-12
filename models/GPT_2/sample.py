"""
用 GPT-2 生成文本: 加载自己训练的 checkpoint 或官方权重。

用法 (在仓库根目录运行):
    python models/GPT-2/sample.py

修改设置: 编辑本文件底部的 SampleConfig(...)。
"""

import os
import pickle
from contextlib import nullcontext
from dataclasses import dataclass

import tiktoken
import torch

from config import GPT2Config
from model import GPT2
from pretrained import load_pretrained


@dataclass
class SampleConfig:
    init_from: str = 'gpt2'             # 'resume': 加载 out_dir/ckpt.pt; 'gpt2' / 'gpt2-medium' / 'gpt2-large' / 'gpt2-xl': 官方权重
    out_dir: str = 'out/gpt2'
    prompt: str = "Hello, I'm a language model,"
    num_samples: int = 3
    max_new_tokens: int = 100
    temperature: float = 0.8            # <1 更保守, >1 更随机
    top_k: int = 200                    # 只在概率最高的 top_k 个 token 中采样
    device: str = 'cuda'
    dtype: str = 'bfloat16'             # 'bfloat16' 或 'float32'
    seed: int = 1337


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """自回归生成: 预测下一个 token, 拼到序列末尾, 重复 max_new_tokens 次。"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]    # 超出上下文长度时只保留最后 block_size 个
        logits, _ = model(idx_cond)                     # (B, 1, vocab_size)
        logits = logits[:, -1, :].float() / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def load_model(cfg):
    """返回 (model, data_dir)。data_dir 用于查找字符级数据集的 meta.pkl。"""
    if cfg.init_from == 'resume':
        checkpoint = torch.load(os.path.join(cfg.out_dir, 'ckpt.pt'), map_location=cfg.device)
        model = GPT2(GPT2Config(**checkpoint['model_config']))
        model.load_state_dict(checkpoint['model'])
        return model, checkpoint['train_config']['data_dir']
    return load_pretrained(cfg.init_from), None


def get_tokenizer(data_dir):
    """返回 (encode, decode)。有 meta.pkl 用字符级映射, 否则用 GPT-2 BPE。"""
    meta_path = os.path.join(data_dir, 'meta.pkl') if data_dir else None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        return (lambda s: [stoi[c] for c in s]), (lambda ids: ''.join(itos[i] for i in ids))
    enc = tiktoken.get_encoding('gpt2')
    return (lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})), enc.decode


def main(cfg):
    torch.manual_seed(cfg.seed)
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[cfg.dtype]
    ctx = nullcontext() if cfg.dtype == 'float32' else torch.autocast(device_type=device_type, dtype=ptdtype)

    model, data_dir = load_model(cfg)
    model.eval().to(cfg.device)
    encode, decode = get_tokenizer(data_dir)

    x = torch.tensor(encode(cfg.prompt), dtype=torch.long, device=cfg.device)[None, :] # (1, T)
    with ctx:
        for _ in range(cfg.num_samples):
            y = generate(model, x, cfg.max_new_tokens, cfg.temperature, cfg.top_k)
            print(decode(y[0].tolist()))
            print('---------------')


if __name__ == '__main__':
    main(SampleConfig())
