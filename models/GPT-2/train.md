# train.py / train_config.py / sample.py / pretrained.py

## 训练流程

```
main()
 ├─ system      seed, TF32, autocast (bf16)
 ├─ model       scratch: 新建 (有 meta.pkl 则用其 vocab_size)
 │              resume:  从 ckpt.pt 恢复结构、权重、优化器、步数
 ├─ optimizer   configure_optimizers() → AdamW
 └─ loop
     ├─ lr = get_lr(iter_num)
     ├─ 每 eval_interval 步: estimate_loss(), val loss 创新低时保存 ckpt.pt
     ├─ 重复 gradient_accumulation_steps 次: forward → loss / steps → backward
     ├─ clip_grad_norm_ → optimizer.step() → zero_grad()
     └─ 每 log_interval 步打印 loss / lr / 耗时 / tok/s
```

一个 iter = 一次参数更新 = `gradient_accumulation_steps` 次前向反向。

## train_config.py

| 分组 | 字段 | 默认 | 说明 |
|---|---|---|---|
| I/O | `out_dir` | `out/gpt2` | checkpoint 目录 |
| | `init_from` | `scratch` | `scratch` / `resume` |
| | `eval_interval` / `eval_iters` | 500 / 100 | 每 500 步评估, 每次平均 100 个 batch |
| | `always_save_checkpoint` | False | False: 只在 val loss 创新低时保存 |
| data | `data_dir` | `data/shakespeare_char` | 含 `train.bin` / `val.bin`, 可选 `meta.pkl` |
| | `batch_size` | 8 | micro-batch, 受显存限制 |
| | `gradient_accumulation_steps` | 8 | |
| AdamW | `learning_rate` | 6e-4 | 峰值 |
| | `weight_decay` | 0.1 | |
| | `beta1` / `beta2` | 0.9 / 0.95 | |
| | `grad_clip` | 1.0 | 0 表示不裁剪 |
| lr | `warmup_iters` / `lr_decay_iters` | 1000 / 20000 | 后者一般等于 `max_iters` |
| | `min_lr` | 6e-5 | 一般为 `learning_rate / 10` |
| system | `dtype` / `compile` | `bfloat16` / False | Windows 上 compile 需 triton-windows |

`block_size` 属于模型结构, 只在 `GPT2Config` 中设置, 避免两处不一致。换机器时一般只改 `batch_size`、`gradient_accumulation_steps`、`dtype`、`compile`。

## get_batch

数据集预处理成一维 token id 数组, 存为 `uint16` 的 `.bin`。随机取 B 个起点, 各截长度 T:

```
data:  ... 15 42  7 99  3 ...
x:         15 42  7 99        输入
y:            42  7 99  3     目标 = 右移一位
```

- `np.memmap`: 按需从磁盘读, 几十 GB 的数据也不占内存。
- `pin_memory()` + `non_blocking=True`: 异步拷到 GPU。循环在 GPU 计算时预取下一个 batch。

## configure_optimizers

### SGD

`loss.backward()` 求出梯度 g, `optimizer.step()` 更新参数: $\theta \leftarrow \theta - \eta g$。
问题: 单个 batch 的梯度噪声大; 所有参数共用步长 η, 而各参数梯度量级差别很大。

### Adam

为每个参数维护两个指数移动平均:

$$
m \leftarrow \beta_1 m + (1-\beta_1) g, \qquad
v \leftarrow \beta_2 v + (1-\beta_2) g^2, \qquad
\theta \leftarrow \theta - \eta \frac{\hat m}{\sqrt{\hat v} + \epsilon}
$$

($\hat m, \hat v$ 是偏差修正, 因为 m、v 从 0 开始。)

- m 是动量, 平均掉噪声。记忆长度约 1/(1-β), β1 = 0.9 约 10 步。
- 除以 √v 把每个参数的步长归一化到 η 量级, 即自适应学习率。GPT 用 β2 = 0.95 (默认 0.999), 反应更快, 训练更稳。
- 代价: 每个参数多存 m、v 两份 fp32, 124M 约多 1 GB 显存。

### Weight decay 与 AdamW

每步把参数往 0 拉一点, 抑制参数过大 (正则化): $\theta \leftarrow \theta - \eta\lambda\theta$。

SGD 中它等价于 loss 加 $\frac{\lambda}{2}\|\theta\|^2$。但在 Adam 中把 λθ 加进梯度, 会被 √v 一起缩放, 梯度大的参数几乎不衰减。AdamW (Loshchilov & Hutter, 2017) 把衰减与梯度更新解耦、单独执行, 对所有参数强度一致。

### 参数分组

| 参数 | 维度 | weight decay |
|---|---|---|
| `Linear.weight`, `Embedding.weight` | 2D | 0.1 |
| `bias`, `LayerNorm.weight` | 1D | 0 |

1D 参数很少 (124M 中约 12 万), 不会过拟合; LayerNorm 的 γ 应在 1 附近, bias 衰减到 0 也无意义。按 `p.dim()` 分组即可。

- 优化器支持参数组, 每组可设不同超参。
- `wte` 与 `lm_head` 共享权重, `model.parameters()` 自动去重。
- `fused=True`: 合并成少数 CUDA kernel, 更快, 结果相同。
- 放在 `train.py` 而非 `model.py`: 优化器属于训练策略, 不属于模型结构。

## get_lr

```
lr │   ╱‾‾‾╲
   │  ╱     ╲
   │ ╱       ╲____  min_lr
   └──────────────── iter
    warmup  cosine
```

- warmup: 起始时参数随机、m/v 不准, 步子太大易发散。
- cosine 衰减: 后期减小步长, 在低谷附近精细收敛。
- 手写函数每步设置 `param_group['lr']`, 比 `lr_scheduler` 更直观。

## 梯度累积

tokens/iter = `batch_size × gradient_accumulation_steps × block_size` = 8 × 8 × 1024 = 65,536。

显存只放得下 8 条, 但大 batch 梯度噪声更小。连续做 8 次前向反向, 梯度在 `p.grad` 上自动累加, 再更新一次。每次 loss 除以 8, 累加结果就是平均梯度, 与一次处理 64 条等价。

## 梯度裁剪

`clip_grad_norm_(params, 1.0)`: 所有梯度拼成一个向量, L2 范数超过 1.0 时等比缩放到 1.0, 方向不变, 防止偶发大梯度一步把参数推坏。

## 混合精度

- `torch.autocast(bfloat16)`: 矩阵乘等用 bf16 计算, 参数和优化器状态仍是 fp32。更快, 激活值显存减半。
- bf16 指数位与 fp32 相同, 范围一样不会溢出, 不需要 fp16 的 `GradScaler`。RTX 30 系列起支持。
- TF32: 剩余 fp32 矩阵乘走 Tensor Core 的 TF32, 精度略降、速度大增。

## Checkpoint

`ckpt.pt` 保存权重、优化器状态、`model_config`、`train_config`、`iter_num`、`best_val_loss`。

- resume 时结构从 checkpoint 读取, 必须一致才能加载权重。
- 必须保存优化器状态, 否则 m、v 归零, 相当于重新开始。
- 保存 `raw_model` (未 compile) 的 state_dict, 否则 key 带 `_orig_mod.` 前缀。
- 124M 的 checkpoint 约 1.5 GB (权重 0.5 + 优化器 1.0)。

## 显存 (124M, block_size 1024, bf16)

| 组成 | 约 |
|---|---|
| 权重 + 梯度 (fp32) | 1 GB |
| AdamW m、v | 1 GB |
| 激活值 | 其余, 随 batch_size 线性增长 |
| batch_size = 8 实测峰值 | 9.3 GiB |

OOM 时减小 `batch_size`, 同比增大 `gradient_accumulation_steps`。

---

## sample.py

```python
for _ in range(max_new_tokens):
    logits = model(idx[:, -block_size:])     # 超长只保留最后 block_size 个
    logits = logits[:, -1, :] / temperature
    # top_k 之外设为 -inf
    idx_next = multinomial(softmax(logits))  # 按概率采样
    idx = cat(idx, idx_next)
```

- 自回归: 每次预测一个 token, 拼回输入再预测下一个。
- `temperature`: T < 1 分布更尖锐 (保守、易重复), T > 1 更平坦 (多样、易胡说)。
- `top_k`: 只在概率最高的 k 个中采样, 截掉低概率长尾。
- tokenizer: 数据目录有 `meta.pkl` 用其 `stoi`/`itos` (字符级), 否则用 tiktoken 的 GPT-2 BPE。
- 每步对整段重新前向, 没有 KV cache, 简单但慢。

## pretrained.py

```python
from pretrained import load_pretrained
model = load_pretrained('gpt2')   # 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'
```

从 Hugging Face `openai-community/<model>` 下载 `model.safetensors`, 逐参数复制。参数命名与 HF 一致, 只有两处差异:

1. HF 用 `Conv1D` 实现 `c_attn`、`c_proj`、`c_fc`, 权重是 (in, out), `nn.Linear` 是 (out, in), 需转置。
2. HF 的 `attn.bias` 是因果 mask 缓存而非参数, 跳过。

官方配置: `vocab_size=50257`, `bias=True`, `block_size=1024`。验证: 官方 124M 对一句普通英文的 loss 约 1.9, 随机初始化约 10.8。

以后做微调, 可在 `train.py` 的 `init_from` 加入 `'gpt2'` 分支调用它。
