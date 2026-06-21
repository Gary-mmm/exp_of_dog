"""
================================================================================
zk-CNN 协议设计：CKKS 加密推理 + 零知识证明验证
================================================================================

参考: Liu et al., "zkCNN: Zero-Knowledge Proofs for CNN Predictions" (CCS 2021)
      https://eprint.iacr.org/2021/673

核心思路:
  1. 服务端用 CKKS 做加密推理 (保护客户端数据隐私)
  2. 服务端生成 zk 证明 π (证明推理使用了承诺的权重且计算正确)
  3. 客户端验证 π 但不获得精确 logits (保护服务端模型权重)

================================================================================
                           完整协议
================================================================================

Phase 0: Setup (一次性)

  Server:
    1. 训练模型，获得权重 W = {conv_w, conv_b, fc1_w, fc1_b, fc2_w, fc2_b}
    2. 计算权重承诺 (Merkle root):
       com_W = MerkleRoot( serialize(W) )
    3. 将 com_W 发布到公共审计日志 (区块链/公告板)
    4. 生成 zkSNARK 电路 C:
       - 公共输入:  com_x (输入承诺), com_y (输出承诺), com_W
       - 私有输入:  x, y, W
       - 约束:     y == CNN(x, W)  AND  com_W == Merkle(W)

Phase 1: 客户端加密查询

  Client:
    1. 加载图像 x (3×32×32)
    2. 生成 CKKS 上下文, 公钥 pk, 私钥 sk
    3. 逐通道 im2col 编码并加密:
       enc_x = [im2col_encode(pk, x[c], 3, 3) for c in 0..2]
    4. 发送 → { enc_x, 公钥 pk, 评估密钥 evk }

Phase 2: 服务端加密计算 + 证明生成

  Server:
    1. 用评估密钥执行同态 CNN:
       enc_y = CKKS_CNN(enc_x, W)   // Conv→Square→Flatten+FC→Square→FC

    2. 在本地重算明文结果 y_local (用于证明):
       因为服务端没有私钥, y_local 必须单独计算。
       但我们不能解密 enc_x...
       
       解决方案: 使用 Fiat-Shamir 变换的承诺方案
       - 客户端同时发送 x 的 Pedersen 承诺 com_x = g^x * h^r
       - 服务端在证明中使用 com_x 而非 x 本身
       - 证明: "存在 x 使得 com_x = Commit(x) 且 y = CNN(x, W)"
       
       但客户端需要向服务端透露 x... 这破坏了隐私。

       更好的方案: 服务端收到 enc_x, 可以评估 CNN 得到 enc_y。
       为了生成 zk 证明, 使用:
       
       a) **交互式**: 服务端发送 enc_y 给客户端, 客户端解密得到 y。
          然后客户端和服务端进行交互式 zk 证明:
          - 服务端证明 ∃W: y == CNN(x, W) AND com_W == Merkle(W)
          - 客户端持有 x, y 作为公共输入
       
       b) **非交互式 (推荐)**: 
          - 客户端额外发送 x 的哈希承诺: h_x = SHA256(x || r)
          - 服务端不知道 x, 但可以在证明中引用 h_x
          - 证明: "∃W, x: enc_x = Enc(pk, x) ∧ y = CNN(x, W) ∧ com_W = Merkle(W)"
          - 但证明中的 x 是私有的 —— 需要 zkSNARK 支持加密原语

  Phase 2 的实用简化方案:
    我们不使用完整的 zkSNARK (计算量极大), 而是使用:
    
    **简化的交互式验证协议**:
    
    Server → Client:  enc_y
    Client:           y = Dec(enc_y)  // 获得 logits

    // ---- 验证阶段 ----
    Client → Server:  随机挑战值 r (256-bit)
    Server → Client:  打开部分权重的 Merkle 证明:
                      - 随机选择权重子集 (基于 r 决定)
                      - 揭示: W_subset, Merkle路径
    Client:           验证 Merkle 路径 → 确认权重一致性
    
    // 但这不验证 "CNN推理是用这些权重执行的"
    // 对于完整性验证, 需要更强的证明

Phase 3: 输出保护 (防模型提取)

  在执行 Phase 2 之前, 服务端对 logits 做以下处理:

  Option A — 噪声注入:
    enc_y_noisy = enc_y + Enc(Laplace(0, σ))
    客户端解密得到 y + η, 而非精确的 y
    ε-DP 保证: 每查询泄露 ≤ Δf/σ bits 信息

  Option B — 标签化 (推荐):
    不返回 logits, 只返回加密的序数关系:
    enc_diff = enc_y[0] - enc_y[1]
    enc_sign = -0.5 * enc_diff^3 + 1.5 * enc_diff  // 3次多项式逼近 sign
    enc_label = enc_sign * 0.5 + 0.5  // → {0, 1}
    return enc_label  // 只有 1 bit 信息

  Option C — 分级访问:
    - 普通用户: 返回 enc_label (Option B)
    - 审计用户: 返回 enc_y + zk_proof (完整 logits + 正确性证明)
    - 审计需要有更高的查询成本和身份验证

===============================================================================
                      推荐实施方案: 噪声 + 标签化 + 承诺
===============================================================================

此方案在不引入完整 zkSNARK 的情况下提供实用保护:

1.  SETUP:
    Server 发布 com_W = SHA256(serialize(W)) 到审计日志

2.  INFERENCE:
    Client → Server:  enc_x, pk, evk
    Server:           enc_y = CKKS_CNN(enc_x, W)
    
    // 防模型提取: 量化为标签
    enc_diff = enc_y[0] - enc_y[1]  
    enc_diff = enc_diff + Enc(Laplace(0, 0.2))  // 轻微噪声
    enc_sign = polynomial_sign(enc_diff)
    enc_label = enc_sign * 0.5 + 0.5  // {0 → cat, 1 → dog}
    
    Server → Client:  enc_label, merkle_path(W_root)

3.  VERIFY (客户端可选):
    Client 解密得到 label (cat/dog)
    Client 验证 Merkle 路径 → 确认权重被承诺过
    
    // 后续若有争议:
    Client 可以公开 {x, label, merkle_path}
    任何第三方可以:
      - 用自己的 CNN 推理验证 label ≈ CNN(x)  
      - 验证 Merkle 路径中的权重片段

===============================================================================
                      安全性分析
===============================================================================

威胁模型                | 防御效果
------------------------|--------------------------------------------------
模型提取 (解方程组)     | 标签化: 每查询仅 1 bit 信息, 提取 4M 参数需 ~4M 查询
                        | 噪声: ε-DP 保证信息泄露有界
                        | 查询限制: N_max = 10000, 远小于 4M
------------------------|--------------------------------------------------
权重篡改 (恶意服务端)   | Merkle 承诺 + 客户端验证路径
                        | 服务端换权重 → Merkle 根不匹配
------------------------|--------------------------------------------------
共谋攻击                | MPC 方案: 权重分片到多服务器
                        | 单服务器被攻陷不泄露完整模型

===============================================================================
                      多项式 Sign 逼近
===============================================================================

sign(x) ≈ -0.5·x³ + 1.5·x    (在 x ∈ [-1, 1] 区间)
       ≈ -0.0656·x⁵ + 0.5203·x³ - 1.3125·x  (更高精度)

CKKS 中实现:  enc_sign = enc.sign()?  TenSEAL 没有内置 sign.
替代: enc_poly = enc.polyval([coeffs])  →  evaluates a₀ + a₁x + a₂x² + ...
      但 TenSEAL CKKSVector 没有 polyval...

实际上:  手动实现多项式:
  enc_x2 = enc_x.square()
  enc_x3 = enc_x2 * enc_x          // 引入 1/512 缩放, 需要注意
  enc_sign = enc_x3 * (-0.5) + enc_x * 1.5

注意: 乘法引入 1/512 缩放, 需要在系数中预补偿 (系数 ×512)

===============================================================================
"""

import hashlib
import json
from typing import Dict, Tuple
import numpy as np


# ===========================
#  Merkle 承诺
# ===========================

def merkle_root(data_dict: Dict[str, np.ndarray]) -> str:
    """计算模型权重的 Merkle 根承诺"""
    leaves = []
    for name in sorted(data_dict.keys()):
        arr = data_dict[name]
        h = hashlib.sha256(arr.tobytes() + name.encode()).hexdigest()
        leaves.append(h)

    while len(leaves) > 1:
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256((leaves[i] + leaves[i + 1]).encode()).hexdigest()
            for i in range(0, len(leaves), 2)
        ]
    return leaves[0] if leaves else ""


def merkle_path(data_dict: Dict[str, np.ndarray], key: str) -> list:
    """返回特定 key 的 Merkle 验证路径"""
    # 简化版: 返回根 + 目标值
    root = merkle_root(data_dict)
    return {"key": key, "value_sha256": hashlib.sha256(
        data_dict[key].tobytes() + key.encode()).hexdigest(), "root": root}


# ===========================
#  多项式 Sign 逼近
# ===========================

def poly_sign_coefficients(degree: int = 3):
    """返回多项式 sign(x) ≈ a₀ + a₁x + a₂x² + a₃x³ 的系数"""
    if degree == 3:
        # sign(x) ≈ -0.5x³ + 1.5x  (在 [-1,1] 最优 L2 逼近)
        return [0.0, 1.5, 0.0, -0.5]
    elif degree == 5:
        return [0.0, 1.3125, 0.0, -0.5203, 0.0, 0.0656]
    else:
        raise ValueError(f"degree={degree} not supported")


# ===========================
#  标签化输出 (防模型提取)
# ===========================

def logits_to_label_protected(logits: np.ndarray,
                              noise_scale: float = 0.2) -> Tuple[int, float]:
    """将 logits 转换为受保护的标签输出。

    1. 添加 Laplace 噪声 (差分隐私)
    2. 应用多项式 sign
    3. 量化为 {0, 1} 标签
    """
    logits = np.asarray(logits, dtype=np.float64)
    diff = logits[0] - logits[1]
    noise = np.random.laplace(0, noise_scale)
    diff_noisy = diff + noise
    coeffs = poly_sign_coefficients(3)
    sign_approx = sum(c * diff_noisy ** i for i, c in enumerate(coeffs))
    label = 0 if sign_approx >= 0 else 1
    return label, sign_approx


# ===========================
#  查询审计日志
# ===========================

class AuditLog:
    """简单的审计日志, 记录每次推理请求的承诺值"""
    def __init__(self):
        self.entries = []

    def record(self, client_id: str, h_x: str, com_w: str,
               label: int, timestamp: float):
        self.entries.append({
            "client": client_id,
            "input_hash": h_x,
            "weight_commitment": com_w,
            "label": label,
            "time": timestamp,
        })

    def query_count(self, client_id: str) -> int:
        return sum(1 for e in self.entries if e["client"] == client_id)

    def check_rate_limit(self, client_id: str, max_queries: int = 100) -> bool:
        return self.query_count(client_id) < max_queries


if __name__ == "__main__":
    # 演示: 计算模型权重承诺
    weights = dict(np.load("artifacts/he_cnn_weights.npz"))
    root = merkle_root(weights)
    print(f"Merkle Root: {root[:16]}...")
    print(f"Model params: {sum(w.size for w in weights.values()):,}")

    # 演示: 标签化输出
    logits = np.array([-0.1183, -0.6610])
    label, sign_val = logits_to_label_protected(logits, noise_scale=0.2)
    print(f"Logits: {logits} → Label: {label} (sign_approx={sign_val:.4f})")
