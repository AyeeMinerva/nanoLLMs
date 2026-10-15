from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

class RotaryEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) 旋转位置编码 (二分折半)
    把位置变成 Q/K 在各二维平面上的旋转。
    """

    def __init__(self, head_size: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_size % 2 == 0, f"head_size must be even, got {head_size}"
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_size, 2).float() / head_size)
        )
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)
        angles = torch.cat([angles, angles], dim=-1)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """前后折半配对，并将每一对旋转对应的角度。"""
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def forward(self, q, k, start_pos: int = 0):
        T = q.size(-2)
        assert start_pos + T <= self.max_seq_len, (
            f"position {start_pos + T} exceeds max_seq_len {self.max_seq_len}"
        )

        cos = self.cos[start_pos:start_pos + T].to(q.dtype)
        sin = self.sin[start_pos:start_pos + T].to(q.dtype)
        return self.apply_rope(q, cos, sin), self.apply_rope(k, cos, sin)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class CausalSelfAttention(nn.Module):
    """多头因果注意力；cache 中保存已经旋转过的 K 和 V。
        RoPE+KV Cache
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_size, config.block_size, config.rope_base)

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        start_pos = 0 if kv_cache is None else kv_cache[0].size(2)
        q, k = self.rope(q, k, start_pos)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # is_causal=True 只适合 query/key 等长；cache 增量前向需要右下角对齐的掩码。
        mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device)
        mask = mask.tril(diagonal=start_pos)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return self.resid_dropout(y), (k, v)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class MiniGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.max_seq_len = config.block_size

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / (2 * config.n_layer) ** 0.5,
                )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_cache=None, return_cache=False, targets=None):
        """无附加参数返回 logits；return_cache 时返回 ``(logits, cache)``。"""
        B, T = idx.size()
        start_pos = 0 if kv_cache is None else kv_cache[0][0].size(2)
        assert start_pos + T <= self.block_size, (
            f"position {start_pos + T} exceeds block_size {self.block_size}"
        )

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        new_cache = []
        for layer, block in enumerate(self.transformer.h):
            layer_cache = None if kv_cache is None else kv_cache[layer]
            x, layer_cache = block(x, layer_cache)
            new_cache.append(layer_cache)

        logits = self.lm_head(self.transformer.ln_f(x))

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            if return_cache:
                return logits, loss, new_cache
            return logits, loss
        if return_cache:
            return logits, new_cache
        return logits

    @staticmethod
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

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, top_k=0, top_p=1.0, temperature=1.0):
        """使用 KV cache 自回归生成，返回形状为 ``(B, T + new_tokens)`` 的 ID。"""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if not torch.is_tensor(prompt_ids):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if prompt_ids.ndim != 2 or prompt_ids.size(1) == 0:
            raise ValueError("prompt_ids must be a non-empty 1-D or 2-D tensor")

        device = next(self.parameters()).device
        ids = prompt_ids.to(device)
        if max_new_tokens == 0:
            return ids
        context = ids[:, -self.block_size:] #截断到最大上下文长度
        logits, cache = self(context, return_cache=True)#首次推理

        for step in range(max_new_tokens):
            next_id = self.sample_next_token(
                logits[:, -1],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            ids = torch.cat([ids, next_id[:, None]], dim=1)
            if step == max_new_tokens - 1:
                break

            if cache[0][0].size(2) < self.block_size:
                logits, cache = self(ids[:, -1:], kv_cache=cache, return_cache=True)
            else:
                logits, cache = self(ids[:, -self.block_size:], return_cache=True)

        return ids


# def load_for_eval(ckpt_path):
#     """读取训练脚本保存的 checkpoint，并返回 ``(model, tokenizer)``。"""
#     from tokenizer import BPETokenizer

#     ckpt_path = Path(ckpt_path)
#     checkpoint = torch.load(ckpt_path, map_location="cpu")
#     config = ModelConfig(**checkpoint["config"])
#     model = MiniGPT(config)
#     model.load_state_dict(checkpoint["model"])

#     tokenizer_path = ckpt_path.parent / "tokenizer.json"
#     tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
#     model.eval()
#     return model, tokenizer
