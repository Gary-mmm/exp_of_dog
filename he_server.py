import base64
import time

import tenseal as ts
from flask import Flask, jsonify, request

from he_context import context_from_bytes
from he_tiny_model import encrypted_inference, load_tiny_weights


app = Flask(__name__)
# 服务启动时加载模型权重，避免每次请求重新读取
MODEL_WEIGHTS = load_tiny_weights()


def b64decode(value):
    """Base64 解码。"""
    return base64.b64decode(value.encode("ascii"))


def b64encode(value):
    """Base64 编码。"""
    return base64.b64encode(value).decode("ascii")


@app.get("/health")
def health():
    """健康检查端点。"""
    return jsonify({"status": "ok", "model": "he_tiny_square_mlp"})


@app.post("/infer")
def infer():
    """密文推理端点。

    接收: JSON { "context": base64(context), "ciphertext": base64(enc_x) }
    返回: JSON { "ciphertext": base64(enc_logits), "server_inference_seconds": float }
    """
    payload = request.get_json(force=True)

    # 反序列化 context (不含私钥) 和密文
    context = context_from_bytes(b64decode(payload["context"]))
    enc_x = ts.ckks_vector_from(context, b64decode(payload["ciphertext"]))

    # 纯密文推理计时
    started = time.perf_counter()
    enc_logits = encrypted_inference(enc_x, MODEL_WEIGHTS)
    elapsed = time.perf_counter() - started

    response = {
        "ciphertext": b64encode(enc_logits.serialize()),
        "server_inference_seconds": elapsed,
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

