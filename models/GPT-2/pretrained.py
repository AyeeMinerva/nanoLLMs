"""加载 OpenAI 官方 GPT-2 权重 (从 Hugging Face 下载)。"""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from config import GPT2Config
from model import GPT2

PRETRAINED_CONFIGS = {
    'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
    'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
    'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
    'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
}

# HF 用 Conv1D 实现这些线性层, 权重形状是 (in, out), nn.Linear 是 (out, in), 需要转置
TRANSPOSED = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')


def load_pretrained(model_type, dropout=0.0):
    config = GPT2Config(**PRETRAINED_CONFIGS[model_type], vocab_size=50257, block_size=1024, bias=True, dropout=dropout)
    model = GPT2(config)
    sd = model.state_dict()

    hf_sd = load_file(hf_hub_download(f'openai-community/{model_type}', 'model.safetensors'))
    loaded = set()
    with torch.no_grad():
        for k, v in hf_sd.items():
            if k.endswith(('.attn.bias', '.attn.masked_bias')): # 因果 mask 缓存, 不是参数
                continue
            k = 'transformer.' + k.removeprefix('transformer.')
            if k.endswith(TRANSPOSED):
                v = v.t()
            assert sd[k].shape == v.shape, f"shape mismatch at {k}: {sd[k].shape} vs {v.shape}"
            sd[k].copy_(v)
            loaded.add(k)

    # lm_head.weight 与 wte.weight 共享, 加载 wte 即可
    missing = set(sd) - loaded - {'lm_head.weight'}
    assert not missing, f"missing keys: {missing}"
    return model
