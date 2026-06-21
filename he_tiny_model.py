from pathlib import Path

import numpy as np
import torch
from torch import nn


# 默认模型参数路径
HE_MODEL_PATH = Path("artifacts/he_tiny_square_mlp.npz")


class HETinySquareMLP(nn.Module):
    """小型 HE 友好模型：客户端 AvgPool 预处理 + 两层全连接 + 平方激活。

    设计思路: 将大核 AvgPool2d(4) 放在客户端侧以明文执行，大幅减小
             加密向量的维度 (192 vs 3072)，降低通信量和密文计算量。
             服务端只需执行两次同态全连接和一次平方激活。

    模型结构:
        Client: AvgPool2d(4,4) → Flatten  → (192)
        Server: fc1: Linear(192→hidden) → Square → fc2: Linear(hidden→2)
    """

    def __init__(self, hidden_size=32):
        super().__init__()
        # 客户端明文预处理：4×4 平均池化 → 3@32×32 → 3@8×8 = 192
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 8 * 8, hidden_size)
        self.square = Square()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.pool(x)         # 明文预处理 (客户端)
        x = self.flatten(x)      # 展平为 192 维
        x = self.fc1(x)          # 全连接 + 平方 + 全连接
        x = self.square(x)
        return self.fc2(x)

    def extract_features(self, x):
        """提取池化+展平后的特征向量，供客户端加密使用。

        Returns: numpy array of shape (batch, 192)
        """
        with torch.no_grad():
            return self.flatten(self.pool(x)).detach().cpu().numpy()



class Square(nn.Module):
    """平方激活函数 f(x) = x^2。"""
    def forward(self, x):
        return x * x


def export_tiny_model(model, path=HE_MODEL_PATH):
    """导出 Tiny 模型权重为 .npz 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fc1_weight=model.fc1.weight.detach().cpu().numpy(),
        fc1_bias=model.fc1.bias.detach().cpu().numpy(),
        fc2_weight=model.fc2.weight.detach().cpu().numpy(),
        fc2_bias=model.fc2.bias.detach().cpu().numpy(),
    )


def load_tiny_weights(path=HE_MODEL_PATH):
    """加载 .npz 权重文件。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run: python train_he_tiny.py"
        )
    return np.load(path)


def load_torch_tiny_model(path=HE_MODEL_PATH):
    """加载权重并重建 PyTorch 模型，供客户端明文预测。"""
    weights = load_tiny_weights(path)
    model = HETinySquareMLP(hidden_size=weights["fc1_bias"].shape[0])
    with torch.no_grad():
        model.fc1.weight.copy_(torch.tensor(weights["fc1_weight"]))
        model.fc1.bias.copy_(torch.tensor(weights["fc1_bias"]))
        model.fc2.weight.copy_(torch.tensor(weights["fc2_weight"]))
        model.fc2.bias.copy_(torch.tensor(weights["fc2_bias"]))
    model.eval()
    return model


def encrypted_inference(enc_x, weights):
    """在密文上执行 Tiny 模型推理：fc1 → square → fc2。

    注意: enc_x 已经是加密的 192 维特征向量（AvgPool+Flatten 由客户端明文完成）。

    Args:
        enc_x: TenSEAL CKKSVector，加密的特征向量
        weights: npz 加载的权重 dict

    Returns:
        enc_logits: 加密的 2 维 logits 向量
    """
    # 矩阵乘法: enc_hidden = enc_x @ fc1_weight.T + fc1_bias
    hidden_matrix = weights["fc1_weight"].T.tolist()
    out_matrix = weights["fc2_weight"].T.tolist()

    enc_hidden = enc_x.matmul(hidden_matrix) + weights["fc1_bias"].tolist()
    enc_hidden = enc_hidden.square()                                           # 平方激活
    enc_logits = enc_hidden.matmul(out_matrix) + weights["fc2_bias"].tolist()  # 输出层
    return enc_logits

