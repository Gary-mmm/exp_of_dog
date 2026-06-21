"""HE-friendly CNN model (3-channel RGB, 16×16 input).

Model architecture:
    Input: 3 x 16 x 16  (client AvgPool2d(2) from 3×32×32 CIFAR-10)
    Conv: 3 → OC, kernel=3, padding=1  →  OC x 16 x 16
    Square activation
    Flatten  →  OC * 256
    FC1:     →  hidden_size
    Square activation
    FC2:     →  2
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn


class HECNNDemo(nn.Module):
    def __init__(self, out_channels: int = 32, hidden_size: int = 512, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, padding=1)
        self.bn_conv = nn.BatchNorm2d(out_channels)
        self.drop_conv = nn.Dropout2d(dropout)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(out_channels * 16 * 16, hidden_size)
        self.bn_fc = nn.BatchNorm1d(hidden_size)
        self.drop_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn_conv(x)
        x = self.drop_conv(x)
        x = x * x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.drop_fc(x)
        x = x * x
        return self.fc2(x)


def fuse_conv_bn(conv_weight, conv_bias, bn_mean, bn_var, bn_weight, bn_bias, eps=1e-5):
    std = (bn_var + eps).sqrt()
    scale = bn_weight / std
    fused_w = conv_weight * scale[:, None, None, None]
    fused_b = conv_bias * scale + (bn_bias - bn_weight * bn_mean / std)
    return fused_w, fused_b


def fuse_linear_bn(lw, lb, bn_mean, bn_var, bn_weight, bn_bias, eps=1e-5):
    std = (bn_var + eps).sqrt()
    scale = bn_weight / std
    fused_w = lw * scale[:, None]
    fused_b = lb * scale + (bn_bias - bn_weight * bn_mean / std)
    return fused_w, fused_b


def export_he_cnn_model(model: HECNNDemo, path: str):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    conv_w, conv_b = fuse_conv_bn(
        sd["conv.weight"], sd["conv.bias"],
        sd["bn_conv.running_mean"], sd["bn_conv.running_var"],
        sd["bn_conv.weight"], sd["bn_conv.bias"],
    )
    fc1_w, fc1_b = fuse_linear_bn(
        sd["fc1.weight"], sd["fc1.bias"],
        sd["bn_fc.running_mean"], sd["bn_fc.running_var"],
        sd["bn_fc.weight"], sd["bn_fc.bias"],
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
        conv_weight=conv_w.numpy(), conv_bias=conv_b.numpy(),
        fc1_weight=fc1_w.numpy(), fc1_bias=fc1_b.numpy(),
        fc2_weight=sd["fc2.weight"].numpy(), fc2_bias=sd["fc2.bias"].numpy(),
    )


def load_he_cnn_weights(path: str) -> Dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python train_he_cnn.py")
    return dict(np.load(p))


def load_torch_he_cnn(path: str) -> HECNNDemo:
    data = load_he_cnn_weights(path)
    oc = data["conv_weight"].shape[0]
    hidden = data["fc1_bias"].shape[0]
    model = HECNNDemo(out_channels=oc, hidden_size=hidden)
    with torch.no_grad():
        model.conv.weight.copy_(torch.tensor(data["conv_weight"]))
        model.conv.bias.copy_(torch.tensor(data["conv_bias"]))
        model.fc1.weight.copy_(torch.tensor(data["fc1_weight"]))
        model.fc1.bias.copy_(torch.tensor(data["fc1_bias"]))
        model.fc2.weight.copy_(torch.tensor(data["fc2_weight"]))
        model.fc2.bias.copy_(torch.tensor(data["fc2_bias"]))
    for m in [model.bn_conv, model.bn_fc]:
        m.weight.data.fill_(1.0); m.bias.data.fill_(0.0)
        m.running_mean.fill_(0.0); m.running_var.fill_(1.0)
    model.eval()
    return model
