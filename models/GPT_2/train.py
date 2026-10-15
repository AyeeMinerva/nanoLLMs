"""
单卡训练 GPT-2。

用法 (在仓库根目录运行):
    python models/GPT-2/train.py

修改训练设置: 编辑本文件底部的 GPT2Config(...) / TrainConfig(...)，或修改 train_config.py 的默认值。
"""

import math
import os
import pickle
import time
from contextlib import nullcontext
from dataclasses import asdict, replace

import numpy as np
import torch

from config import GPT2Config
from model import GPT2
from config import TrainConfig


def get_batch(data_dir, split, batch_size, block_size, device):
    """从 {split}.bin 中随机截取 batch_size 段长度为 block_size 的序列。"""
    # 每次重新创建 memmap, 避免内存泄漏
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]) # 目标是右移一位的序列
    if 'cuda' in device:
        # pin_memory + non_blocking: CPU->GPU 异步拷贝
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """创建 AdamW: 2D 参数 (矩阵, embedding) 做 weight decay, 1D 参数 (bias, LayerNorm) 不做。"""
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    no_decay_params = [p for p in params if p.dim() < 2]
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    print(f"decayed params: {sum(p.numel() for p in decay_params):,}, "
          f"non-decayed params: {sum(p.numel() for p in no_decay_params):,}")
    return torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, fused=(device_type == 'cuda'))


def get_lr(it, cfg):
    """学习率调度: 线性 warmup -> cosine 衰减 -> 保持 min_lr。"""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 从 1 平滑下降到 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, cfg, block_size, ctx):
    """在 train / val 上各取 eval_iters 个 batch, 返回平均 loss。"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(cfg.data_dir, split, cfg.batch_size, block_size, cfg.device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main(model_config, cfg):
    # ---------------- system ----------------
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.fp32_precision = 'tf32' # 允许 fp32 矩阵乘使用 TF32 (torch>=2.9 的新 API)
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[cfg.dtype]
    ctx = nullcontext() if cfg.dtype == 'float32' else torch.autocast(device_type=device_type, dtype=ptdtype)

    # ---------------- model ----------------
    iter_num, best_val_loss = 0, float('inf')
    checkpoint = None
    if cfg.init_from == 'resume':
        ckpt_path = os.path.join(cfg.out_dir, 'ckpt.pt')
        print(f"resuming from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        model_config = GPT2Config(**checkpoint['model_config']) # 模型结构必须与 checkpoint 一致
        iter_num, best_val_loss = checkpoint['iter_num'], checkpoint['best_val_loss']
    else:
        # 若数据集提供了 meta.pkl (如字符级数据集), 用它的词表大小
        meta_path = os.path.join(cfg.data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                vocab_size = pickle.load(f)['vocab_size']
            print(f"found vocab_size = {vocab_size} in {meta_path}")
            model_config = replace(model_config, vocab_size=vocab_size)

    model = GPT2(model_config).to(cfg.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    print(f"number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = configure_optimizers(model, cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # 释放内存

    raw_model = model # 保存 checkpoint 时用未编译的模型, 避免 state_dict 的 key 带 '_orig_mod.' 前缀
    if cfg.compile:
        print("compiling the model...")
        model = torch.compile(model)

    block_size = model_config.block_size
    tokens_per_iter = cfg.batch_size * cfg.gradient_accumulation_steps * block_size
    print(f"tokens per iteration: {tokens_per_iter:,}")

    # ---------------- training loop ----------------
    x, y = get_batch(cfg.data_dir, 'train', cfg.batch_size, block_size, cfg.device)
    t0 = time.time()
    while True:
        lr = get_lr(iter_num, cfg) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 评估 + 保存 checkpoint
        if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters:
            losses = estimate_loss(model, cfg, block_size, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if iter_num > 0 and (losses['val'] < best_val_loss or cfg.always_save_checkpoint):
                best_val_loss = min(best_val_loss, losses['val'])
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_config': asdict(model_config),
                    'train_config': asdict(cfg),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {cfg.out_dir}")
                torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt.pt'))
                checkpoint = None
        if iter_num >= cfg.max_iters:
            break

        # 前向 + 反向: 梯度累积模拟更大的 batch
        for _ in range(cfg.gradient_accumulation_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accumulation_steps # 累积的是平均梯度
            x, y = get_batch(cfg.data_dir, 'train', cfg.batch_size, block_size, cfg.device) # GPU 计算时预取下一个 batch
            loss.backward()

        # 参数更新
        if cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # 日志
        t1 = time.time()
        dt, t0 = t1 - t0, t1
        if iter_num % cfg.log_interval == 0:
            loss_val = loss.item() * cfg.gradient_accumulation_steps # .item() 会同步 CPU 与 GPU
            print(f"iter {iter_num}: loss {loss_val:.4f}, lr {lr:.2e}, "
                  f"time {dt * 1000:.0f}ms, {tokens_per_iter / dt:,.0f} tok/s")
        iter_num += 1


if __name__ == '__main__':
    model_config = GPT2Config()   # GPT-2 124M
    train_config = TrainConfig()
    main(model_config, train_config)
