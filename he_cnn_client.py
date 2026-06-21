"""Client for CKKS-encrypted CNN inference (3-channel RGB 16×16).

Usage:
    python he_cnn_client.py --server http://127.0.0.1:5001 --samples 10
"""

import argparse, base64, json, statistics
import requests, tenseal as ts, torch, numpy as np
from torch import nn
from he_cnn_model import load_torch_he_cnn
from he_cnn_ops import im2col_encode_channel
from he_context import create_ckks_context, public_context_bytes
from train_catdog import build_loaders


def b64encode(value):
    return base64.b64encode(value).decode("ascii")


def json_size(payload):
    return len(json.dumps(payload).encode("utf-8"))


def run(args):
    _, test_loader = build_loaders(args.data_dir, batch_size=1, num_workers=0)
    model = load_torch_he_cnn(args.model)
    context = create_ckks_context()
    pub_ctx = public_context_bytes(context)

    total = correct = 0
    abs_diffs, server_times, traffic_bytes = [], [], 0

    for image, label in test_loader:
        if total >= args.samples:
            break

        # Client: 3×32×32 → 3×16×16
        image = nn.functional.avg_pool2d(image, 2)
        image_np = image.squeeze(0).numpy()  # [3, 16, 16]

        label_val = int(label.item())
        with torch.no_grad():
            plain_logits = model(image).squeeze(0).numpy()
        plain_pred = int(plain_logits.argmax())

        # Encrypt: per-channel im2col encoding (3×3 kernel, padding=1)
        enc_channels = []
        windows_nb = None
        for c in range(3):
            enc, wn = im2col_encode_channel(context, image_np[c], 3, 3, 1, padding=1)
            enc_channels.append(enc)
            if windows_nb is None:
                windows_nb = wn

        payload = {
            "context": b64encode(pub_ctx),
            "ciphertexts": [b64encode(ct.serialize()) for ct in enc_channels],
            "windows_nb": windows_nb,
        }
        traffic_bytes += json_size(payload)

        resp = requests.post(f"{args.server}/infer", json=payload, timeout=args.timeout)
        resp.raise_for_status()
        result = resp.json()
        traffic_bytes += json_size(result)

        enc_logits = ts.ckks_vector_from(context, base64.b64decode(result["ciphertext"]))
        he_logits = enc_logits.decrypt()
        he_pred = int(max(range(len(he_logits)), key=lambda idx: he_logits[idx]))

        correct += int(he_pred == label_val)
        total += 1
        server_times.append(float(result["server_inference_seconds"]))
        abs_diffs.extend(abs(float(a) - float(b)) for a, b in zip(plain_logits, he_logits))

        print(f"sample={total:03d} label={label_val} "
              f"plain={plain_pred} he={he_pred} "
              f"time={server_times[-1]:.2f}s")

    acc = correct / total if total else 0.0
    print(f"\n=== CKKS MLaaS CNN (3ch RGB) ===")
    print(f"samples={total}  accuracy={acc*100:.1f}%")
    print(f"avg_server_time={statistics.mean(server_times):.2f}s" if server_times else "")
    print(f"avg_logit_diff={statistics.mean(abs_diffs):.6f}" if abs_diffs else "")
    print(f"traffic={traffic_bytes/1024/1024:.1f}MB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://127.0.0.1:5001")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--model", default="artifacts/he_cnn_weights.npz")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--timeout", type=float, default=600.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
