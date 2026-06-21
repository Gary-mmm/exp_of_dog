import argparse

import numpy as np
import torch
from torch import nn

from he_tiny_model import HETinySquareMLP, export_tiny_model
from train_catdog import build_loaders


def evaluate(model, loader, device):
    """评估 Tiny 模型的测试集损失和准确率。"""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32,
                        help="隐藏层维度 (默认 32)")
    parser.add_argument("--output", default="artifacts/he_tiny_square_mlp.npz")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(2026)
    np.random.seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, 0)

    model = HETinySquareMLP(hidden_size=args.hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = 0
        loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)

        test_loss, test_acc = evaluate(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={loss_sum / seen:.4f} "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc * 100:.2f}% "
            f"best={best_acc * 100:.2f}%"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    export_tiny_model(model.cpu(), args.output)
    print(f"Saved HE tiny model to: {args.output}")


if __name__ == "__main__":
    main()

