import argparse
import base64
import json
import statistics

import requests
import tenseal as ts
import torch

from he_context import create_ckks_context, public_context_bytes
from he_tiny_model import load_torch_tiny_model
from train_catdog import build_loaders


def b64encode(value):
    """Base64 编码。"""
    return base64.b64encode(value).decode("ascii")


def encoded_json_size(payload):
    """计算 JSON 序列化后的字节数 (用于统计通信量)。"""
    return len(json.dumps(payload).encode("utf-8"))


def plain_predict(model, image):
    """明文推理：返回 logits numpy 数组。"""
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
    return logits.squeeze(0).numpy()


def encrypt_feature_vector(context, model, image):
    """提取图像特征向量并使用 CKKS 加密。

    客户端先执行 AvgPool2d(4) + Flatten 得到 192 维明文向量，
    再用 CKKS 加密，大幅减少密文通信量。
    """
    features = model.extract_features(image.unsqueeze(0))[0]
    return ts.ckks_vector(context, features.tolist())


def run(args):
    """端到端 CKKS MLaaS 评估主函数。

    流程:
        1. 加载测试集和模型
        2. 创建 CKKS 上下文和公钥序列化
        3. 对每个样本:
            a. 明文预测 (作为基准)
            b. 加密特征向量发送给服务端
            c. 接收密文 logits 并解密
            d. 比较明文/密文预测一致性
        4. 统计并打印四项指标
    """
    _, test_loader = build_loaders(args.data_dir, batch_size=1, num_workers=0)
    model = load_torch_tiny_model(args.model)
    context = create_ckks_context()
    public_context = public_context_bytes(context)  # 不含私钥的序列化

    total = 0
    correct = 0
    abs_diffs = []          # 记录每个样本的 |plain_logit - he_logit|
    server_times = []       # 记录服务端密文推理耗时
    traffic_bytes = 0       # 累加总通信字节数

    for image, label in test_loader:
        if total >= args.samples:
            break

        image = image.squeeze(0)
        label_value = int(label.item())

        # 明文推理基准
        plain_logits = plain_predict(model, image)
        plain_pred = int(plain_logits.argmax())

        # 加密并构造请求
        enc_x = encrypt_feature_vector(context, model, image)
        request_payload = {
            "context": b64encode(public_context),
            "ciphertext": b64encode(enc_x.serialize()),
        }
        traffic_bytes += encoded_json_size(request_payload)

        # 发送请求到 Flask 服务端
        response = requests.post(
            f"{args.server}/infer",
            json=request_payload,
            timeout=args.timeout,
        )
        response.raise_for_status()
        result = response.json()
        traffic_bytes += encoded_json_size(result)

        # 解密服务端返回的密文 logits
        enc_logits = ts.ckks_vector_from(context, base64.b64decode(result["ciphertext"]))
        he_logits = enc_logits.decrypt()
        he_pred = int(max(range(len(he_logits)), key=lambda idx: he_logits[idx]))

        correct += int(he_pred == label_value)
        total += 1
        server_times.append(float(result["server_inference_seconds"]))
        abs_diffs.extend(abs(float(a) - float(b)) for a, b in zip(plain_logits, he_logits))

        print(
            f"sample={total:03d} label={label_value} "
            f"plain_pred={plain_pred} he_pred={he_pred} "
            f"server_time={server_times[-1]:.4f}s"
        )

    # 汇总统计
    accuracy = correct / total if total else 0.0
    avg_abs_diff = statistics.mean(abs_diffs) if abs_diffs else 0.0
    avg_server_time = statistics.mean(server_times) if server_times else 0.0
    traffic_mb = traffic_bytes / (1024 * 1024)

    print()
    print("=== End-to-end CKKS MLaaS Evaluation ===")
    print(f"samples: {total}")
    print(f"end_to_end_traffic_mb: {traffic_mb:.4f}")
    print(f"avg_server_inference_seconds: {avg_server_time:.4f}")
    print(f"avg_abs_plain_he_logit_diff: {avg_abs_diff:.8f}")
    print(f"he_accuracy: {accuracy * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:5000")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model", default="artifacts/he_tiny_square_mlp.npz")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

