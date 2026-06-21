"""Flask server for CKKS-encrypted CNN inference.

Receives per-channel im2col-encoded ciphertexts and runs:
  Conv(1→OC) → Square → Flatten → FC1 → Square → FC2 → 2

Usage:
    python he_cnn_server.py
"""

import base64
import json
import time

import tenseal as ts
from flask import Flask, jsonify, request

from he_cnn_model import load_he_cnn_weights
from he_cnn_ops import encrypted_cnn_inference
from he_context import context_from_bytes

app = Flask(__name__)
# 服务启动时加载模型权重
MODEL = load_he_cnn_weights("artifacts/he_cnn_weights.npz")
OC = MODEL["conv_weight"].shape[0]            # 输出通道数
KH, KW = MODEL["conv_weight"].shape[2], MODEL["conv_weight"].shape[3]  # 卷积核尺寸


def _b64d(val):
    return base64.b64decode(val.encode("ascii"))


def _b64e(val):
    return base64.b64encode(val).decode("ascii")


@app.get("/health")
def health():
    """健康检查端点：返回模型基本信息。"""
    return jsonify({
        "status": "ok",
        "model": "he_cnn_rgb",
        "out_channels": OC,
        "kernel": [KH, KW],
    })


@app.post("/infer")
def infer():
    """密文 CNN 推理端点。

    接收: JSON { "context": base64(ctx), "ciphertexts": [base64(ch0), ...], "windows_nb": int }
    返回: JSON { "ciphertext": base64(logits), "server_inference_seconds": float }

    每个输入通道对应一个 im2col 编码的 CKKS 密文向量。
    """
    payload = request.get_json(force=True)
    ctx = context_from_bytes(_b64d(payload["context"]))
    windows_nb = int(payload["windows_nb"])

    # 逐通道反序列化 im2col 密文
    enc_channels = [
        ts.ckks_vector_from(ctx, _b64d(ct))
        for ct in payload["ciphertexts"]
    ]

    # 纯密文 CNN 推理: Conv → Square → Flatten → FC1 → Square → FC2
    started = time.perf_counter()
    enc_logits = encrypted_cnn_inference(
        enc_channels, windows_nb,
        MODEL["conv_weight"], MODEL["conv_bias"],
        MODEL["fc1_weight"], MODEL["fc1_bias"],
        MODEL["fc2_weight"], MODEL["fc2_bias"],
    )
    elapsed = time.perf_counter() - started

    return jsonify({
        "ciphertext": _b64e(enc_logits.serialize()),
        "server_inference_seconds": elapsed,
    })


@app.route("/model_info")
def model_info():
    """返回模型结构信息，供客户端参考。"""
    return jsonify({
        "architecture": "Conv(1→OC,3×3)→Square→Flatten→FC(OC·225→hidden)→Square→FC→2",
        "input": "1-channel grayscale 32×32 (client converts RGB→gray)",
        "kernel_rows": KH,
        "kernel_cols": KW,
        "out_channels": OC,
        "weights_file": "artifacts/he_cnn_weights.npz",
    })


if __name__ == "__main__":
    print(f"HE CNN server (RGB) starting  OC={OC}  kernel={KH}×{KW}")
    app.run(host="127.0.0.1", port=5001, debug=False)