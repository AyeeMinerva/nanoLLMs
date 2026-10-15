"""四种解码策略：greedy / temperature / top-k / top-p。"""

import torch


def sample_next_token(logits, top_k=0, top_p=1.0, temperature=1.0):
    """logits: (B, vocab_size)，返回 (B,) 个 token ID。"""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("top_k must be >= 0 and top_p must be in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1)  # greedy，无需除以 0

    logits = logits / temperature
    if top_k > 0:
        cutoff = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if top_p < 1:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        # 超过阈值的词要删掉，但保留第一个越过阈值的词及概率最高的词。
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(1, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)  # 截断之后重新归一化
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
