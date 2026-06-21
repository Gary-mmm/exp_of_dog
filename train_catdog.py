import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# CIFAR-10 原始标签: 3=cat, 5=dog
CAT_LABEL = 3
DOG_LABEL = 5


class Square(nn.Module):
    """平方激活函数 f(x)=x^2，替代 ReLU 用于密文计算。

    ReLU 是分段函数 max(0,x)，在密文域中需要比较操作，CKKS 不支持。
    平方函数仅需一次乘法，是密文友好的非线性替代方案。
    """
    def forward(self, x):
        return x * x


class CatDogSquareCNN(nn.Module):
    """2 conv + 2 fc CNN using HE-friendly square activations and AvgPool.

    BatchNorm is used during training and fused into the preceding Conv/Linear
    weights at export time so that the HE inference graph remains purely
    linear + square + pool.

    模型结构:
        Input:  (3, 32, 32)  RGB 图像，像素范围 [0, 1]
        Conv1:  Conv2d(3→32, k=5, p=2) → BN → Square → AvgPool(2)
                → (32, 16, 16)
        Conv2:  Conv2d(32→64, k=5, p=2) → BN → Square → AvgPool(2)
                → (64, 8, 8)
        FC1:    Flatten → Linear(4096→256) → BN → Square → (256)
        FC2:    Linear(256→2) → (2)  logits
    """

    def __init__(self):
        super().__init__()
        # features 部分：两层卷积
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)             # 训练辅助，导出时熔合
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # classifier 部分：两层全连接
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)     # 64*8*8 = 4096
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 2)               # 输出 cat/dog 两个 logit

    def forward(self, x):
        # 前向传播：conv → bn → square → pool
        x = self.pool1(self.bn1(self.conv1(x)) ** 2)
        x = self.pool2(self.bn2(self.conv2(x)) ** 2)
        x = self.flatten(x)
        x = (self.bn_fc(self.fc1(x))) ** 2
        return self.fc2(x)                         # 输出层无激活，直接 logits


class CatDogDataset(Dataset):
    """从 CIFAR-10 中筛选 cat/dog 类并映射为二分类标签。

    CIFAR-10 标签映射:
        cat (3) → 0
        dog (5) → 1
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # 只保留 cat 和 dog 对应的样本索引
        self.indices = [
            idx
            for idx, (_, label) in enumerate(dataset)
            if label in (CAT_LABEL, DOG_LABEL)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        binary_label = 0 if label == CAT_LABEL else 1   # cat→0, dog→1
        return image, binary_label


def build_loaders(data_dir, batch_size, num_workers):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.ToTensor()

    train_base = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_base = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_set = CatDogDataset(train_base)
    test_set = CatDogDataset(test_base)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return loss_sum / total, correct / total


def fuse_conv_bn(conv_weight, conv_bias, bn_running_mean, bn_running_var, bn_weight, bn_bias, eps=1e-5):
    """将 BatchNorm2d 熔合到前面的 Conv2d 权重中。

    数学推导:
        BN(x) = gamma * (x - mu) / sqrt(var + eps) + beta
        Conv(x) = W * x + b

    熔合后:
        fused_W = W * (gamma / sigma)           ← 按通道缩放卷积核
        fused_b = b * (gamma / sigma) + (beta - gamma * mu / sigma)

    熔合后 Conv-BN 等价为单个 Conv，推理图不再需要 BN 层，
    这对密文推理至关重要——CKKS 无法执行 BN 中的 sqrt/除法。
    """
    std = (bn_running_var + eps).sqrt()
    scale = bn_weight / std
    fused_weight = conv_weight * scale[:, None, None, None]
    fused_bias = conv_bias * scale + (bn_bias - bn_weight * bn_running_mean / std) if conv_bias is not None else (bn_bias - bn_weight * bn_running_mean / std)
    return fused_weight, fused_bias


def fuse_linear_bn(linear_weight, linear_bias, bn_running_mean, bn_running_var, bn_weight, bn_bias, eps=1e-5):
    """将 BatchNorm1d 熔合到前面的 Linear 权重中。

    原理同 fuse_conv_bn，但 Linear 权重为 2D 矩阵，scale 广播维度不同。
    """
    std = (bn_running_var + eps).sqrt()
    scale = bn_weight / std
    fused_weight = linear_weight * scale[:, None]
    fused_bias = linear_bias * scale + (bn_bias - bn_weight * bn_running_mean / std)
    return fused_weight, fused_bias


def export_parameters(model, output_dir):
    """导出 BN 熔合后的模型参数为 .pt 和 .npz 格式。

    导出流程:
        1. 提取 state_dict
        2. 对 conv1, conv2, fc1 分别执行 BN 熔合
        3. fc2 无 BN，直接导出
        4. 保存 PyTorch 格式和 NumPy 格式
        5. 生成 manifest.json 记录模型元信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # BN 熔合：将训练时的 BN 参数吸收进相邻的 Conv/Linear 权重
    conv1_w, conv1_b = fuse_conv_bn(
        sd["conv1.weight"], sd["conv1.bias"],
        sd["bn1.running_mean"], sd["bn1.running_var"],
        sd["bn1.weight"], sd["bn1.bias"],
    )
    conv2_w, conv2_b = fuse_conv_bn(
        sd["conv2.weight"], sd["conv2.bias"],
        sd["bn2.running_mean"], sd["bn2.running_var"],
        sd["bn2.weight"], sd["bn2.bias"],
    )
    fc1_w, fc1_b = fuse_linear_bn(
        sd["fc1.weight"], sd["fc1.bias"],
        sd["bn_fc.running_mean"], sd["bn_fc.running_var"],
        sd["bn_fc.weight"], sd["bn_fc.bias"],
    )
    fc2_w = sd["fc2.weight"]
    fc2_b = sd["fc2.bias"]

    fused_state = {
        "conv1.weight": conv1_w.numpy(),
        "conv1.bias": conv1_b.numpy(),
        "conv2.weight": conv2_w.numpy(),
        "conv2.bias": conv2_b.numpy(),
        "fc1.weight": fc1_w.numpy(),
        "fc1.bias": fc1_b.numpy(),
        "fc2.weight": fc2_w.numpy(),
        "fc2.bias": fc2_b.numpy(),
    }

    torch.save(fused_state, output_dir / "catdog_square_cnn_state_dict.pt")
    np.savez(output_dir / "catdog_square_cnn_weights_biases.npz", **fused_state)

    manifest = {
        "classes": {"0": "cat", "1": "dog"},
        "input": "CIFAR-10 RGB image, shape [3, 32, 32], pixel range [0, 1]",
        "architecture": [
            "conv1: Conv2d(3,32,k=5,s=1,p=2) + BN(fused) -> square -> AvgPool2d(2)",
            "conv2: Conv2d(32,64,k=5,s=1,p=2) + BN(fused) -> square -> AvgPool2d(2)",
            "flatten -> fc1: Linear(4096,256) + BN(fused) -> square",
            "fc2: Linear(256,2)",
        ],
        "parameter_files": {
            "pytorch": "catdog_square_cnn_state_dict.pt",
            "numpy": "catdog_square_cnn_weights_biases.npz",
        },
        "parameter_names": list(fused_state.keys()),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, test_loader = build_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )

    model = CatDogSquareCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # CosineAnnealingLR: 学习率按余弦曲线从 lr 衰减到接近 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)     # 更高效的梯度清零
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            # 梯度裁剪：防止平方激活导致的梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        scheduler.step()
        train_loss = running_loss / seen
        test_loss, test_acc = evaluate(model, test_loader, device)

        # 保存测试集上的最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc * 100:.2f}% "
            f"best={best_acc * 100:.2f}%"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    export_parameters(model, Path(args.output_dir))
    print(f"Saved best model parameters to: {Path(args.output_dir).resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./artifacts")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
