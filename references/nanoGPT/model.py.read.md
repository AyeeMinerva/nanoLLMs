# model.py阅读笔记
## LayerNorm 类
```python
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```
此类功能：对输入的每个 token 特征向量做 LayerNorm 标准化，并在需要时用可学习的 weight 和 bias 再进行缩放和平移。

公式：
$$
\mathrm{LayerNorm}(x)_i
=
\mathrm{weight}_i \cdot \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
+
\mathrm{bias}_i
$$
其中：
$$
\mu = \frac{1}{ndim}\sum_{j=1}^{ndim} x_j
$$

$$
\sigma^2 = \frac{1}{ndim}\sum_{j=1}^{ndim}(x_j-\mu)^2
$$

ndim(number of dimensions):特征数

weight:可学习的缩放参数

bias:偏置项

其中weight和bias是可学习参数。

---

> Q&A: **YOLO中使用的是BN(BatchNorm)，GPT使用的是LN(LayerNorm) 为什么？**

首先明确YOLO中的BN的使用方式。BN是对整个Batch进行归一化。一整个batch的样本图片的张量一起经过前面的卷积层，得到这一层的输出向量，再一起对这个张量[B,C,H,W]的各个通道c，每个通道内算均值和方差，归一化然后再进入下一层。训练是以batch为单位，整个batch一起处理，而不是逐个图片处理。

然后是YOLO为什么不逐个图片进行Norm（也就是InstanceNorm/样本级通道归一化），而是多个图片一起算均值。在图片数据集内光照复杂的情况下，InstanceNorm确实可以减弱光照导致的风格差异，但是在检测中，这反而会去除掉可能很有用的比如明暗度、对比度信息，有些信息就是要依赖于光照强度来判别的，都归一化之后就体现不出相对于基准全局的光照强度了。

对于LN，GPT一次前向传播处理整个输入的序列张量[B,T,C]，LN会对这个张量的每个token对应的部分分别做归一化。

---

> Q&A: **为什么使用一个$\epsilon$？**

因为在计算标准化时，某些输入的方差可能非常小，甚至接近 0。如果直接用标准差作为分母，就可能出现除零或数值不稳定的问题。

加入 `epsilon` 以后：

$$
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

它的作用主要有三点：

1. 防止分母为 0。
2. 提高训练过程的数值稳定性。
3. 避免当方差过小时，归一化结果被过度放大。

在本类中 `F.layer_norm(..., eps=1e-5)` 中，`1e-5` 就是这个稳定项。

---

## CausalSelfAttention类

- B = batch size，批大小，一次喂给模型多少条样本
- T = time steps / sequence length，序列长度，一条样本里有多少个 token
- C = channel / embedding dimension，特征维度，也就是每个 token 的向量长度，等于 `n_embd`

> Q&A: **为什么还要过c_proj这个线性层？**

经过多头注意力机制部分中的QKV之后，张量里还是对每个头处理的部分分块的 如[head1]|[head2]|[head3]。经过c_proj这个线性层之后，可以将多头的输出都混合起来。

## MLP类

---
> Q&A: **dropout有什么用，为什么现在很少有用了？**

dropout有什么用：可以防止过拟合，提高泛化。当在模型较大，数据较少的情况下，使用dropout很合适。

在没有dropout的情况下，某些神经元可能过度依赖于方差较大，幅度更高的特征；或者某神经元学到的是错误的误判某个噪声特征，其后续神经元可能会专门演化出极大的负权重抵消它；或者许多神经元的输出都依赖于某个“明星”神经元。施加dropout，由于每个神经元都有可能随时被归零，迫使每个神经元必须提取更独立通用的特征，降低了对特定输入组合的敏感度。

此外每次训练迭代时，dropout随机遮盖一部分神经元，相当于是在训练一个更小的子网络。最终推理时，相当于取了所有模型的平均预测，可以显著提升模型的泛化能力。
---

## GPT类

nn.Embedding(num_embeddings, embedding_dim):

是一个形状为(num_embeddings, embedding_dim)的可学习权重矩阵weight, 传入一个整数张量如[1,3],它直接输出矩阵的第一行和第三行组合成输出

---
> Q&A: **Transformer 的注意力矩阵看起来是按网格位置排列的，模型究竟是在“死记位置对应关系”，还是真的理解了“词义关联”？如果输入变成“爱吃苹果皮”，泛化能力会崩溃吗？**

**问题的引出与直觉困惑**

在注意力机制中，加权聚合过程表现为矩阵乘法：

$$\text{attn\_weights} = \begin{bmatrix}
a_{\text{我}\to\text{我}} & a_{\text{我}\to\text{爱}} & a_{\text{我}\to\text{吃}} & a_{\text{我}\to\text{苹果}} \\
a_{\text{爱}\to\text{我}} & a_{\text{爱}\to\text{爱}} & a_{\text{爱}\to\text{吃}} & a_{\text{爱}\to\text{苹果}} \\
a_{\text{吃}\to\text{我}} & a_{\text{吃}\to\text{爱}} & a_{\text{吃}\to\text{吃}} & a_{\text{吃}\to\text{苹果}} \\
a_{\text{苹果}\to\text{我}} & a_{\text{苹果}\to\text{爱}} & a_{\text{苹果}\to\text{吃}} & a_{\text{苹果}\to\text{苹果}}
\end{bmatrix}, \quad
\begin{bmatrix}
\mathbf{o}_{\text{我}} \\
\mathbf{o}_{\text{爱}} \\
\mathbf{o}_{\text{吃}} \\
\mathbf{o}_{\text{苹果}}
\end{bmatrix} = \text{attn\_weights} \times V$$

对于第 4 行的“苹果”，其更新向量为：
$$\mathbf{o}_{\text{苹果}} = a_{\text{苹果}\to\text{我}} \mathbf{v}_{\text{我}} + a_{\text{苹果}\to\text{爱}} \mathbf{v}_{\text{爱}} + a_{\text{苹果}\to\text{吃}} \mathbf{v}_{\text{吃}} + a_{\text{苹果}\to\text{苹果}} \mathbf{v}_{\text{苹果}}$$

形式上这个矩阵操作严格绑定在坐标槽位 $(i, j)$ 上。这让人容易产生疑虑：模型是否只是在死记“位置 4 去抓取位置 3 的信息”，而根本没理解“苹果”与“吃”的语义？如果插入词变成“爱吃苹果皮”，原本对应位置关系被打破，泛化能力是否会崩溃？

**底层机理：注意力权重由内容内积动态生成，而非槽位记忆**

矩阵中的权重并不是固定的网络参数，而是由 Query 与 Key 在运行时实时计算的点积结果：

$$a_{i \to j} \propto \mathbf{q}_i \cdot \mathbf{k}_j$$

以“苹果”（位置 4）匹配“吃”（位置 3）为例，向量由内容嵌入 $\mathbf{e}$ 与位置编码 $\mathbf{p}$ 叠加后经线性投影得到：
* $\mathbf{q}_{\text{苹果}} = (\mathbf{e}_{\text{苹果}} + \mathbf{p}_4) W_Q$
* $\mathbf{k}_{\text{吃}} = (\mathbf{e}_{\text{吃}} + \mathbf{p}_3) W_K$

将两者的点积展开为四项：

$$\mathbf{q}_{\text{苹果}} \cdot \mathbf{k}_{\text{吃}} = \underbrace{\mathbf{e}_{\text{苹果}} W_Q W_K^T \mathbf{e}_{\text{吃}}^T}_{\text{① 纯语义匹配 (Content-to-Content)}} + \underbrace{\mathbf{e}_{\text{苹果}} W_Q W_K^T \mathbf{p}_3^T}_{\text{② 内容到位置 (Content-to-Position)}} + \underbrace{\mathbf{p}_4 W_Q W_K^T \mathbf{e}_{\text{吃}}^T}_{\text{③ 位置到内容 (Position-to-Content)}} + \underbrace{\mathbf{p}_4 W_Q W_K^T \mathbf{p}_3^T}_{\text{④ 纯几何位置偏置 (Position-to-Position)}}$$

* **项 ① 是决定性的基本盘**：只要在几何嵌入空间中，“及物动词”与“可食用名词”被映射到高内积方向，无论词在第 3 位还是第 10 位，纯语义内积都会自然给出高分。
* **项 ②、③、④ 是调节项**：仅提供距离和顺序的几何先验，防止跨越整段文本产生无关的语义误连，但它并不主导语义判定。

**场景拓展：换成“爱吃苹果皮”，模型如何泛化？**

当输入变成 `[我, 爱, 吃, 苹果, 皮]` 时，模型依赖以下两个机制保持稳健泛化：

1. **多头注意力解耦分工**：模型拥有数十个并行的 Head。某些局部注意力头专注于修饰关系，让“皮”聚合“苹果”的特征，演化为“苹果的外皮”；而全局语义头专注于动宾支配，“吃”跨过“苹果”直接将注意力落在中心词“皮”上。
2. **多层抽象**：在网络浅层，词与词结合为短语单位（“苹果皮”）；在网络高层，每个位置的向量已脱离孤立字面，变为富含上下文的抽象概念。

**现代演进：相对位置编码（如 RoPE）抹平绝对槽位依赖**

现代主流 LLM（如 LLaMA、Qwen、DeepSeek）普遍采用**旋转位置编码（RoPE）**：

$$\mathbf{q}_m \cdot \mathbf{k}_n = g(\mathbf{e}_m, \mathbf{e}_n, m - n)$$

它在数学保证了点积大小严格只取决于词义向量本身与**两者的相对距离差 $(m - n)$**。只要“吃”与目标词的相对结构保持相似，无论整句话出现在文章开头还是末尾，位置调制项完全恒定，彻底消除了对固定矩阵槽位的硬性依赖。

---

> Q&A: **什么是旋转位置编码（RoPE）？为什么现代大模型几乎全面放弃了传统的绝对位置编码而转向它？**

旋转位置编码（Rotary Position Embedding，简称 RoPE）是苏剑林等人提出的一种将位置信息注入 Transformer 的机制。它的核心目标是：**在实现上采用绝对位置编码的形式（计算轻量、易于并行），但在注意力点积运算中严格涌现出相对位置的数学性质。**

传统的绝对位置编码（如可学习位置编码、正弦/余弦绝对编码）采用的是“向量相加”：$\mathbf{x}_m = \mathbf{e}_m + \mathbf{p}_m$。这种方式存在明显的缺陷：
1. **语义特征与几何坐标杂糅**：直接做向量平移可能破坏词嵌入原有的流形结构，且展开点积时包含繁杂的内容-位置交互杂项。
2. **缺乏严格的相对距离感知**：无法在代数层面保证“相距 1 个词的两个 Token”在句子开头和句子末尾具有完全一致的几何内积偏置。
3. **外推与长上下文扩展困难**：直接加绝对坐标很难泛化到训练长度之外的位置。

RoPE 摒弃了“加法”，改用**复数旋转（乘法）**：
在二维子空间中，为每个位置分配一个旋转矩阵 $\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$。
- 位置 $m$ 处的 Query 向量逆时针旋转 $m\theta$ 角度：$\mathbf{q}_m = \mathbf{R}(m\theta)\mathbf{q}$
- 位置 $n$ 处的 Key 向量逆时针旋转 $n\theta$ 角度：$\mathbf{k}_n = \mathbf{R}(n\theta)\mathbf{k}$

当两者计算注意力内积时，利用正交旋转矩阵相乘等价于角度相减的性质：
$$\mathbf{q}_m \cdot \mathbf{k}_n = (\mathbf{R}(m\theta)\mathbf{q})^T (\mathbf{R}(n\theta)\mathbf{k}) = \mathbf{q}^T \mathbf{R}(m\theta)^T \mathbf{R}(n\theta) \mathbf{k} = \mathbf{q}^T \mathbf{R}\big((n - m)\theta\big) \mathbf{k}$$
这一步消除了绝对位置标号 $m$ 与 $n$，**使内积结果严格仅取决于 Query 与 Key 之间的相对距离 $(n - m)$ 以及两者的语义夹角**。

在高维向量中，RoPE 通过**分块正交旋转**实现：将 $d$ 维向量拆分成 $d/2$ 个二维子空间，每个子空间赋予不同的旋转基频 $\theta_i = b^{-2i/d}$。高频维度负责捕捉紧邻词的局部语法结构，低频维度负责捕捉长距离的篇章级依赖。

此外，RoPE 具备天然的**长程衰减特性**（距离越远，内积上界自然呈现衰减），且其线性的旋转性质使得大模型在后期进行上下文拓展（如通过 Position Interpolation、NTK-aware、YaRN 进行 32K/128K 扩窗）时极为简单且代价极低。因此，LLaMA、Qwen、DeepSeek、Mistral 等现代大语言模型几乎全量采用了该技术。

---