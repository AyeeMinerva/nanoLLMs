# mini-GPT 模型笔记

对应代码：`model.py`。结构同 `GPT_2/model.py`，新增 `RotaryEmbedding`，Attention / Block / MiniGPT 支持 KV cache。

config 使用 `ModelConfig`，字段包括 `vocab_size, block_size, n_layer, n_head, n_embd, dropout, bias, rope_base`（`rope_base` 默认 10000）。

# 第一部分：RoPE

## 1. 为什么需要它

GPT-2 用 `wpe = nn.Embedding(block_size, n_embd)` 学一个绝对位置向量，加到词向量上。问题在于注意力分数 `q·k` 里混进了绝对位置，模型很难直接学到"相隔多远"这种信息，而且超过 `block_size` 的位置根本没有向量可用。

RoPE 换了个思路：不加向量，而是把 Q 和 K **按位置旋转**一个角度。

## 2. 核心思想

把 `head_size` 维向量拆成 `hs/2` 对，每对看作二维平面上的一个点。位置 `m` 的第 `i` 对旋转角度 `m·θ_i`：

```
θ_i = base^(-2i / hs),   base = 10000

[x1']   [cos mθ  -sin mθ] [x1]
[x2'] = [sin mθ   cos mθ] [x2]
```

关键性质：两个旋转后向量的点积只与**相对位置**有关

```
<R(m)q, R(n)k> = <q, R(n-m)k>
```

因为旋转矩阵满足 `R(m)ᵀR(n) = R(n-m)`。所以注意力分数自然编码了 `n - m`，不需要额外参数。

不同的 `i` 频率不同：低维转得快（捕捉近距离），高维转得慢（捕捉远距离），类似时钟的秒针和时针。

### 2.1 转得快和转得慢

"低维 / 高维"指第几对维度（下标 `i`），不是向量维数。每对有自己的角速度 `θ_i`，表示每往后挪一个词元多转多少弧度。`i` 越大，`θ_i` 越小，转得越慢。

以 `hs = 64`（32 对）为例：

| 第 i 对 | θ_i（弧度/词元） | 一圈要几个词元 (2π/θ_i) |
|---|---|---|
| 0 | 1.0 | ≈ 6 |
| 8 | 0.1 | ≈ 63 |
| 16 | 0.01 | ≈ 628 |
| 24 | 0.001 | ≈ 6283 |
| 31 | 0.00013 | ≈ 47000 |

位置 `m` 处的角度是 `m · θ_i`，就是代码里 `torch.outer(pos, inv_freq)` 算出的表。

### 2.2 为什么要快慢搭配

- **只有快针**：相邻词元很好区分，但约 6 个词元就转满一圈，距离 1 和距离 7 几乎分不清。
- **只有慢针**：远距离不绕圈，但相邻词元之间角度几乎不变，分不清"紧挨着"和"隔了 3 个"。
- **快慢搭配**：像钟表的秒针、分针、时针一起读，或者像二进制的低位和高位。转速按几何级数排开，周期覆盖约 6 到约 47000 个词元，任何距离都有几根针的转速正好合适。

直观对比：

- 绝对位置编码（GPT-2 `wpe`）：给每个位置贴一张学出来的标签，标签之间没有预设的几何关系。
- RoPE：一组转速不同的时钟。比较两个位置时，看同一根针在两处的夹角 `(n-m)·θ_i`。

### 2.3 在注意力分数里的体现

```
q·k = Σ_i |q_i| |k_i| · cos((n - m)·θ_i + φ_i)
```

`φ_i` 是 `q_i` 与 `k_i` 本身的夹角，只和内容有关。每对维度贡献一个以相对距离为自变量、频率为 `θ_i` 的余弦波，相当于对"距离"做傅里叶分解。模型想关注近邻，就把能量放在快针（小 `i`）上；想关注远处，就用慢针（大 `i`）。这种分配是学出来的，RoPE 只提供多尺度的刻度。

### 2.4 base 的作用

`base` 决定最慢那根针有多慢，它的周期应远大于上下文长度，否则远距离又会绕圈混淆。长上下文模型常把 `base` 调大（如 50 万、100 万），就是让所有针整体变慢。

想看具体数字：打印 `RotaryEmbedding` 里的 `inv_freq` 和 `2 * math.pi / inv_freq`。

## 3. 配对方式：前后折半

两种常见配对，选一种并贯彻到底：

| 方式 | 配对 | 代表实现 |
|---|---|---|
| 相邻两维 | `(x0,x1), (x2,x3), ...` | 原论文、Meta 原版 LLaMA |
| 前后折半 | `(x0,x_{hs/2}), (x1,x_{hs/2+1}), ...` | HF LLaMA、本仓库 |

本实现用前后折半，因为写成向量运算最简单（`RotaryEmbedding.apply_rope`）：

```python
x1, x2 = x.chunk(2, dim=-1)
x_rotated = torch.cat([-x2, x1], dim=-1)   # [x1, x2] -> [-x2, x1]
return x * cos + x_rotated * sin
```

展开一下，前半 `x1' = x1·cos - x2·sin`，后半 `x2' = x2·cos + x1·sin`，正好是上面的旋转矩阵。为此 `cos/sin` 表要把 `hs/2` 个角度复制两份拼成 `hs` 维（代码里的 `torch.cat([angles, angles])`）。

两种方式数学上等价（只是维度重排），但训练和推理必须用同一种。

## 4. 代码走读

```
__init__:
  inv_freq  (hs/2,)       θ_i
  pos       (T_max,)      0..max_seq_len-1
  angles    (T_max, hs/2) 外积 m·θ_i
  angles    (T_max, hs)   复制两份，匹配前后折半
  cos, sin  buffer        预计算，查表即可

apply_rope(x, cos, sin):
  对单个张量做旋转

forward(q, k, start_pos):
  取 cos[start_pos : start_pos+T]，对 q 和 k 各调用一次 apply_rope
```

- `register_buffer(..., persistent=False)`：不是可训练参数，会跟着 `model.to(device)` 走，但不写进 `state_dict`（可以从超参数重算）。
- `.to(q.dtype)`：混合精度训练时保持 dtype 一致。
- `start_pos`：当前片段在整个序列里的起始位置。全量前向时为 0；KV cache 增量解码时等于已缓存的长度。

## 5. 常见坑

- **只旋转 Q/K，不旋转 V**。位置信息只需要进入注意力分数 `q·k`，V 是被加权的内容。
- **增量解码时位置要接着算**。生成第 `t` 个词时它的位置是 `t`，不是 0。传错 `start_pos`，开/关 cache 的 logits 就对不上。
- **cache 里存的是旋转后的 K**。历史 K 已经按各自位置旋转过，不要重复旋转。
- `head_size` 必须是偶数。

# 第二部分：Causal Self-Attention + KV cache

## 1. 和 GPT-2 版本的差异

结构沿用 `GPT_2/model.py` 的 `CausalSelfAttention`，只改三处：

| | GPT-2 | mini-GPT |
|---|---|---|
| 位置信息 | 在模型入口加 `wpe` | 在注意力内部对 Q/K 做 RoPE |
| 历史 K/V | 每次重算 | 可传入 `kv_cache` 复用 |
| 因果掩码 | `is_causal=True` | 显式 `attn_mask` |
| Attention / Block 返回值 | `y` | `(y, new_cache)` |
| 顶层 forward | `forward(idx, targets)` | `forward(idx, kv_cache, return_cache, targets)` |
| logits | 无 targets 时只返回最后一个位置 | 总是返回全部位置 |

## 2. 接口

```python
attn = CausalSelfAttention(config)   # 需要 n_embd, n_head, dropout, bias, block_size, rope_base
y, cache = attn(x)                   # 全量前向，x: (B, T, C)
y, cache = attn(x_new, kv_cache=cache)  # 增量前向，x_new 通常是 (B, 1, C)
```

`kv_cache = (k, v)`，形状都是 `(B, nh, T_past, hs)`。注意力层总是返回新 cache，要不要保留由模型决定（机制与策略分离）。

## 3. 前向流程

```
x (B,T,C)
 └─ c_attn ─> q, k, v           各 (B,nh,T,hs)
 └─ start_pos = T_past          无 cache 时为 0
 └─ RoPE(q, k, start_pos)       只旋转新词元的 q/k
 └─ k = cat(past_k, k, dim=2)   在序列维拼接 -> (B,nh,T_past+T,hs)
    v = cat(past_v, v, dim=2)
 └─ SDPA(q, k, v, mask)         (B,nh,T,hs)
 └─ c_proj ─> y (B,T,C)
```

顺序很重要：**先 RoPE，再拼 cache**。历史 K 在当初计算时已经按自己的位置旋转过了，cache 里存的就是旋转后的结果。

## 4. 为什么 KV cache 成立

自回归生成第 `t` 步时，前 `t-1` 个词元的 K、V 和上一步算出来的完全一样（因果掩码保证它们看不到未来，输入不变，输出也不变）。所以只需要：

- 为新词元算 1 个 q/k/v
- 把新 k/v 追加到 cache
- 用 1 个 q 对全部 `t` 个 k/v 做注意力

每层每步的投影计算从 `O(t)` 个词元降到 1 个，代价是显存里多存 `2 × n_layer × B × nh × t × hs` 个数。

## 5. 掩码为什么不用 `is_causal=True`

`is_causal=True` 生成的是**左上角对齐**的下三角，等价于 `ones(T_q, T_k).tril(0)`。当 `T_q == T_k` 时没问题；增量解码时 `T_q=1, T_k=t`，它会变成 `[1, 0, 0, ...]`，新词元只能看到第 0 个位置，结果错误。

正确的掩码是**右下角对齐**：绝对位置 `start_pos + i` 的 query 能看到位置 `j ≤ start_pos + i` 的 key：

```python
attn_mask = torch.ones(T, T_total, dtype=torch.bool).tril(diagonal=start_pos)
```

例子，`start_pos=2, T=2`（缓存 2 个，新来 2 个）：

```
         k0 k1 k2 k3
q(pos2) [ 1  1  1  0 ]
q(pos3) [ 1  1  1  1 ]
```

`start_pos=0` 时退化为普通下三角，所以训练和推理共用一行代码。bool 掩码中 `True` 表示允许注意。

## 6. 自检思路（kv_cache_equivalence）

同一个序列：
1. 一次性全量前向，取每个位置的输出
2. 逐词元增量前向，每步传入上一步的 cache

两者每个位置的输出应在 `1e-4` 内一致。对不上时按顺序排查：`start_pos` 是否等于历史长度 → 掩码是否右下角对齐 → 拼接维度是否是 `dim=2` → 是否重复旋转了历史 K。

## 7. MiniGPT：把每层 cache 串起来

```python
logits, loss = model(idx, targets=targets)                          # 训练
logits, cache = model(prompt, return_cache=True)                    # 预填充 prompt
logits, cache = model(next_id, kv_cache=cache, return_cache=True)   # 每步生成 1 个
```

- 模型级 cache 是 list，第 `i` 个元素是第 `i` 层的 `(k, v)`。
- 顶层从 `kv_cache[0][0].size(2)` 读出历史长度，只用来检查不超过 `block_size`；每层注意力自己从 cache 推出 `start_pos`。
- logits 总是返回全部位置 `(B, T, vocab_size)`，困惑度评估和 cache 等价性检查都要用每个位置的输出。生成时取 `logits[:, -1, :]`。
- `generate(prompt_ids, max_new_tokens, top_k, top_p, temperature)` 实现上述增量循环；采样规则独立写在 `sampling.py`。cache 达到 `block_size` 时重新预填充最近一段上下文。

## 8. 注意

- 位置上限是 `config.block_size`（RoPE 预计算表的长度），超过会触发 assert。`generate` 中到上限时重算最近 `block_size` 个位置；超过训练上下文时生成质量未必稳定。
