import tenseal as ts


def create_ckks_context():
    """Create a CKKS context with enough depth for CNN inference.

    The HE CNN pipeline needs up to 5 multiplications (conv → square →
    fc1 → square → fc2).  With poly_modulus_degree=16384 we have 8192
    slots which comfortably holds the im2col-encoded 16×16 image.

    coeff_mod_bit_sizes: 7 个模数构成 6 级乘法链，每个 40-bit prime 支持
    一次同态乘法后的 rescale。首尾 60-bit 用于容纳 scale 和噪声。
    global_scale = 2^40: 决定 CKKS 浮点编码的精度。
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
    )
    context.global_scale = 2**40
    context.generate_relin_keys()    # 重线性化密钥：乘法后降维密文
    context.generate_galois_keys()   # 伽罗瓦密钥：支持向量旋转 (im2col 卷积必需)
    return context


def public_context_bytes(context):
    """序列化不含私钥的上下文，用于发送给服务端。

    服务端可执行密文计算（公钥加密、重线性化、旋转），但无法解密。
    """
    return context.serialize(
        save_public_key=True,
        save_secret_key=False,       # 服务端不可解密用户数据
        save_galois_keys=True,
        save_relin_keys=True,
    )


def private_context_bytes(context):
    """序列化包含私钥的完整上下文，仅客户端本地保留。"""
    return context.serialize(
        save_public_key=True,
        save_secret_key=True,
        save_galois_keys=True,
        save_relin_keys=True,
    )


def context_from_bytes(data):
    """从序列化字节流恢复 TenSEAL 上下文对象。"""
    return ts.context_from(data)
