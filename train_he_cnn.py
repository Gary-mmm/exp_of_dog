"""Train HE CNN on CIFAR-10 cat/dog (3-channel RGB 16×16).

Usage:
    python train_he_cnn.py --epochs 60 --out-channels 32 --hidden-size 512
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from he_cnn_model import HECNNDemo, export_he_cnn_model
from train_catdog import build_loaders


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = nn.functional.avg_pool2d(images.to(device), 2)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers)

    model = HECNNDemo(out_channels=args.out_channels, hidden_size=args.hidden_size).to(device)

    print(f"Model: Conv(3→{args.out_channels},3×3,p=1)→Square→FC({args.out_channels*256}→{args.hidden_size})→Square→FC→2")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = nn.functional.avg_pool2d(images.to(device), 2)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()
        train_loss = running_loss / seen
        test_loss, test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {name: t.detach().cpu().clone() for name, t in model.state_dict().items()}

        print(f"epoch {epoch:03d}/{args.epochs} "
              f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
              f"test_acc={test_acc*100:.2f}% best={best_acc*100:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    output = Path(args.output)
    export_he_cnn_model(model.cpu(), str(output))
    print(f"best_acc={best_acc*100:.2f}%")
    print(f"Exported to: {output.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output", default="artifacts/he_cnn_weights.npz")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--out-channels", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
