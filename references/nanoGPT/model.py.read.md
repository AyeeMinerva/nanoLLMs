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

> Q&A: **YOLO中使用的是BN(BatchNorm)，GPT使用的是LN(LayerNorm) 为什么？**

首先明确YOLO中的BN的使用方式。BN是对整个Batch进行归一化。一整个batch的样本图片的张量一起经过前面的卷积层，得到这一层的输出向量，再一起对这个张量[B,C,H,W]的各个通道c，每个通道内算均值和方差，归一化然后再进入下一层。训练是以batch为单位，整个batch一起处理，而不是逐个图片处理。

然后是YOLO为什么不逐个图片进行Norm（也就是InstanceNorm/样本级通道归一化），而是多个图片一起算均值。在图片数据集内光照复杂的情况下，InstanceNorm确实可以减弱光照导致的风格差异，但是在检测中，这反而会去除掉可能很有用的比如明暗度、对比度信息，有些信息就是要依赖于光照强度来判别的，都归一化之后就体现不出相对于基准全局的光照强度了。

对于LN，GPT一次前向传播处理整个输入的序列张量[B,T,C]，LN会对这个张量的每个token对应的部分分别做归一化。

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



## CausalSelfAttention类

- B = batch size，批大小，一次喂给模型多少条样本
- T = time steps / sequence length，序列长度，一条样本里有多少个 token
- C = channel / embedding dimension，特征维度，也就是每个 token 的向量长度，等于 `n_embd`

> "为什么还要过c_proj这个线性层？"
经过多头注意力机制部分中的QKV之后，张量里还是对每个头处理的部分分块的。经过c_proj这个线性层之后，可以将多头的输出都混合起来。

