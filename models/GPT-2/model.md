# model.py / config.py

## 总览

```
idx (B, T)                                  token id
 ├─ wte(idx) + wpe(pos)        (B, T, C)    词嵌入 + 位置嵌入
 ├─ dropout
 ├─ Block × n_layer            (B, T, C)
 │    x = x + attn(ln_1(x))                 token 之间交换信息
 │    x = x + mlp(ln_2(x))                  每个 token 独立加工
 ├─ ln_f
 └─ lm_head                    (B, T, V)    每个位置对下一个 token 的打分
```

| 记号 | 含义 | config 字段 |
|---|---|---|
| B | batch size | - |
| T | 序列长度, T ≤ block_size | `block_size` |
| C | 每个 token 的向量维度 | `n_embd` |
| V | 词表大小 | `vocab_size` |
| nh / hs | 头数 / 每头维度, hs = C / nh | `n_head` |

## config.py

| 字段 | 默认 (124M) | 说明 |
|---|---|---|
| `block_size` | 1024 | 最大上下文长度, 即 `wpe` 的行数 |
| `vocab_size` | 50304 | 50257 补到 64 的倍数, 矩阵乘更快; 多出的 token 不会出现, 模型会学到给它们极低概率 |
| `n_layer` | 12 | Block 数 |
| `n_head` | 12 | 必须整除 `n_embd` |
| `n_embd` | 768 | C |
| `dropout` | 0.0 | 预训练用 0, 微调可试 0.1 |
| `bias` | True | Linear / LayerNorm 是否带 bias, GPT-2 原版为 True |

`__post_init__` 在实例创建后自动执行, 提前拦截非法配置。

## MLP

```
(B, T, C) → c_fc → (B, T, 4C) → GELU → c_proj → (B, T, C) → dropout
```

注意力在 token 之间搬运信息, MLP 对每个 token 的向量独立做非线性加工。GELU(x) = x·Φ(x), Φ 为标准正态 CDF, 是平滑版 ReLU。

## CausalSelfAttention

```
x (B, T, C)
 → c_attn → (B, T, 3C) → split → q, k, v 各 (B, T, C)
 → view + transpose → (B, nh, T, hs)              拆成 nh 个头
 → scaled_dot_product_attention(is_causal=True)
 → transpose + view → (B, T, C)                   拼回
 → c_proj → dropout
```

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{hs}} + M\right)V
$$

- $QK^T$ 是 (T, T) 矩阵, (i, j) 表示 token i 对 token j 的关注程度。
- 除以 $\sqrt{hs}$: 点积方差随 hs 线性增长, 不缩放会使 softmax 过于尖锐、梯度消失。
- $M$ 是因果 mask, 上三角为 $-\infty$, token i 只能看到 j ≤ i。于是每个位置都只凭过去预测下一个 token, 一条长度 T 的序列同时提供 T 个训练样本。
- `F.scaled_dot_product_attention` 把整个公式融合成一个 kernel (FlashAttention), 不显式构造 (T, T) 矩阵, 更快更省显存。
- `c_attn` 一个 Linear 同时算出 Q/K/V, 等价于三个 Linear, 一次矩阵乘更快。
- `transpose` 后内存不连续, `view` 前要 `.contiguous()`。

> Q: 为什么还要过 `c_proj`?
>
> 多头拼接后是 `[head1 | head2 | ...]`, 各段互不相关。`c_proj` 把各头结果线性混合, 再写回残差流。

## Block

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

- 残差连接: 主干 x 称为残差流 (residual stream), 每个子层只往上"加"修改。反向传播时梯度沿 `+` 直通底层, 深层网络才训得动。
- Pre-LN: LayerNorm 放在子层之前, 残差流本身不归一化。比原始 Transformer 的 Post-LN `LN(x + f(x))` 更稳定, 现代 LLM 基本都用 Pre-LN。

## GPT2

### 嵌入

`nn.Embedding(V, C)` 是 (V, C) 的可学习矩阵, `wte(idx) == wte.weight[idx]`, 即按 id 取行。

- `wte`: token 嵌入, "是什么词"。
- `wpe`: 可学习的绝对位置嵌入, "在第几位"。两者直接相加。

### 权重共享 (weight tying)

```python
self.transformer.wte.weight = self.lm_head.weight
```

两层共用矩阵 E (V × C):

- 输入: 查表取第 i 行 $E_i$。
- 输出: `nn.Linear` 计算 $y = xW^T$, 令 $W = E$ 得 $\mathrm{logit}_i = h \cdot E_i$, 即最终向量与每个词向量的点积, 越相似分数越高。

"id → 向量"与"向量 → 与各词的相似度"天然可共用一个矩阵。E 约 3860 万参数, 共享省掉 124M 中近 1/3。

> Q: 为什么不用逆矩阵?
>
> E 是 V × C 非方阵, 不存在逆。输出层也不是要"还原"输入, 而是给每个词打分再 softmax 分类, 点积正好给出这个分数。

### 初始化

- 所有 Linear / Embedding 权重: $\mathcal{N}(0, 0.02^2)$, bias 为 0。
- `c_proj.weight`: 标准差再除以 $\sqrt{2 \cdot n\_layer}$。

每个 Block 往残差流加 2 次, 共 2·n_layer 次。n 个独立随机量相加方差变为 n 倍, 把写回残差流的 `c_proj` 标准差缩小 $\sqrt{2 \cdot n\_layer}$ 倍, 初始化时残差流方差就与层数无关。

`self.apply(fn)` 对每个子模块递归调用 fn。

### forward

- 训练 (`targets` 非 None): 对全部 T 个位置算 logits, 与右移一位的 `targets` 算交叉熵。
- 推理 (`targets` 为 None): 只需下一个 token, 只对最后一个位置算 `lm_head`。用 `[-1]` 而非 `-1` 以保留 T 维, 输出 (B, 1, V)。

$$
\mathcal{L} = -\frac{1}{BT}\sum_{b,t} \log p(\text{target}_{b,t})
$$

随机初始化时每个词概率约 1/V, loss ≈ ln(50304) ≈ 10.8, 是训练起点的参考值。

---

# Q&A

## LayerNorm

对每个 token 的 C 维向量单独标准化, 再缩放平移:

$$
\mathrm{LN}(x)_i = \gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i,
\quad \mu = \frac{1}{C}\sum_j x_j,
\quad \sigma^2 = \frac{1}{C}\sum_j (x_j - \mu)^2
$$

- γ (`weight`, 初始 1)、β (`bias`, 初始 0) 可学习。
- ε (1e-5) 防止方差接近 0 时除零或结果被过度放大。
- torch ≥ 2.1 的 `nn.LayerNorm` 支持 `bias=False`; nanoGPT 自定义 `LayerNorm` 类只为兼容旧版, 这里直接用 `nn.LayerNorm`。

## 为什么 CNN (如 YOLO) 用 BatchNorm, GPT 用 LayerNorm

| | BatchNorm | LayerNorm |
|---|---|---|
| 统计范围 | batch 内所有样本的同一通道 | 单个 token 的所有通道 |
| 依赖 batch | 是, 推理用训练期滑动平均 | 否, 训练推理一致 |

- CNN 中同一通道在不同图片、位置上表示同一种特征, 跨 batch 统计有意义。不逐图归一化 (InstanceNorm), 是因为那会抹掉亮度、对比度等全局信息, 检测有时正依赖它们。
- 序列中不同位置含义各异、长度不一; 自回归推理时 batch 常为 1, BN 无法工作。LN 只看单个 token, 与 batch 和长度无关。

## dropout 的作用, 为什么大模型预训练不用

训练时以概率 p 把输出随机置 0 (其余乘 1/(1-p) 保持期望)。

- 防止共适应: 神经元不能依赖某个"明星"神经元, 被迫学习更独立通用的特征。
- 隐式集成: 每步训练一个随机子网络, 推理时关闭 dropout, 相当于对所有子网络取平均。

大模型预训练数据量远大于模型容量, 每条数据往往只见一次, 主要矛盾是欠拟合, dropout 只会降低学习效率。数据少的微调才可能用。

## 注意力在记位置, 还是理解语义

注意力矩阵按 (i, j) 排列, 但它不是固定参数, 而是每次由内容实时计算: $a_{i \to j} \propto q_i \cdot k_j$。

以"我爱吃苹果"中"苹果"(位置 4) 关注"吃"(位置 3) 为例, 输入是内容嵌入 e 加位置嵌入 p:

$$
q_{苹果} \cdot k_{吃}
= \underbrace{e_{苹果} W_Q W_K^T e_{吃}^T}_{内容-内容}
+ \underbrace{e_{苹果} W_Q W_K^T p_3^T}_{内容-位置}
+ \underbrace{p_4 W_Q W_K^T e_{吃}^T}_{位置-内容}
+ \underbrace{p_4 W_Q W_K^T p_3^T}_{位置-位置}
$$

- 内容-内容项是语义匹配: 只要动词与可食用名词被学成高内积方向, 不论处在哪个位置都得高分。
- 其余三项提供距离和顺序先验。

换成"我爱吃苹果皮"也不会崩: 多个头分工 (有的让"皮"关注"苹果"组成"苹果皮", 有的让"吃"关注宾语中心"皮"); 多层堆叠后, 高层向量已是融合上下文的抽象表示。

## RoPE (旋转位置编码)

GPT-2 "加法"式绝对位置嵌入的缺点: 内容与位置混杂; 相同相对距离不保证相同的位置项; 无法外推到 `block_size` 之外。

RoPE (Su et al., 2021) 改用"乘法": q、k 每两维看作一个平面, 位置 m 旋转 mθ:

$$
q_m = R(m\theta)\, q, \quad k_n = R(n\theta)\, k, \quad
R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}
$$

由 $R(a)^T R(b) = R(b - a)$:

$$
q_m \cdot k_n = q^T R\big((n - m)\theta\big)\, k
$$

点积只依赖相对距离 n - m。高维时拆成 C/2 个平面, 各用频率 $\theta_i = 10000^{-2i/d}$: 高频维度捕捉相邻词, 低频维度捕捉长距离依赖。RoPE 也便于扩展上下文 (Position Interpolation、NTK-aware、YaRN), LLaMA、Qwen、DeepSeek、Mistral 都用它。
