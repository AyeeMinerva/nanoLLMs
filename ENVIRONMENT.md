# 环境

| 项目 | 值 |
|---|---|
| 环境名 | `nanollm` (conda) |
| 路径 | `C:\Users\Ayee\.conda\envs\nanollm` |
| Python | 3.12 |
| PyTorch | 2.9.0+cu128 |
| 依赖 | [`requirements.txt`](requirements.txt) |
| 测试机器 | RTX 5070 Ti Laptop 12GB, Windows |

## 使用

```powershell
conda activate nanollm
cd F:\Projects\nanoLLMs

python models/GPT-2/train.py    # 训练
python models/GPT-2/sample.py   # 生成文本
```

下载 Hugging Face 权重需要代理 (或改用镜像):

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
# 或: $env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 重建环境

conda:

```powershell
conda create -n nanollm python=3.12 -y
conda activate nanollm
pip install -r requirements.txt
```

uv:

```powershell
uv sync --locked
# torch 固定从 pyproject.toml 配置的 PyTorch cu128 源安装；无需另装 CUDA Toolkit。
```

## 验证

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
# 期望输出: 2.9.0+cu128 True True
# 当前机器也已验证 RTX 5070 Ti Laptop 上的 CUDA 矩阵乘法。
```
