import math

import tenseal as ts


def almost_equal(vec1, vec2, tol=1e-3):
    """判断两个向量是否近似相等 (相对误差和绝对误差双重容忍)。"""
    return all(math.isclose(a, b, rel_tol=tol, abs_tol=tol) for a, b in zip(vec1, vec2))


def demo_ckks_basic():
    """基础 CKKS 演示：加密、同态加法、同态乘法、解密校验。

    实验 1: 明文 X=[1.5,2.5], Y=[3.4,4.5]
            → 加密为 Enc(X), Enc(Y)
            → 同态计算 Enc(X)+Enc(Y) 和 Enc(X)×Enc(Y)
            → 解密验证结果与明文直接运算是否一致
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],     # 3 级乘法链
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()

    x = [1.5, 2.5]
    y = [3.4, 4.5]

    enc_x = ts.ckks_vector(context, x)
    enc_y = ts.ckks_vector(context, y)

    enc_add = enc_x + enc_y           # 同态加法
    enc_mul = enc_x * enc_y           # 同态乘法 (消耗一级乘法深度)

    dec_add = enc_add.decrypt()
    dec_mul = enc_mul.decrypt()

    plain_add = [a + b for a, b in zip(x, y)]
    plain_mul = [a * b for a, b in zip(x, y)]

    print("=== Part 1: Basic CKKS Demo ===")
    print(f"Plaintext X: {x}")
    print(f"Plaintext Y: {y}")
    print(f"Decrypted Enc(X) + Enc(Y): {dec_add}")
    print(f"Plaintext X + Y:           {plain_add}")
    print(f"Decrypted Enc(X) * Enc(Y): {dec_mul}")
    print(f"Plaintext X * Y:           {plain_mul}")
    print(f"Addition check passed: {almost_equal(dec_add, plain_add)}")
    print(f"Multiplication check passed: {almost_equal(dec_mul, plain_mul)}")
    print()


def experiment_bfv_repeated_self_multiplication(max_rounds=12):
    """
    BFV 连续自乘实验：验证同态乘法噪声增长。

    设计: 选择明文 [1]，理论上平方后仍是 [1]，一旦解密结果偏离即
          说明噪声预算耗尽。BFV 没有 rescale 机制，噪声随每次乘法
          呈指数级增长。

    预期: 约 4-5 轮后出现失真。
    """
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=8192,
        plain_modulus=1032193,             # 明文模数 (素数)
    )
    context.generate_relin_keys()

    ciphertext = ts.bfv_vector(context, [1])
    expected = [1]
    failure_round = None

    print("=== Part 2: BFV Noise Budget Experiment ===")
    print("Plaintext seed: [1]")
    print("Expected plaintext after each self-multiplication: [1]")

    for round_idx in range(1, max_rounds + 1):
        try:
            ciphertext = ciphertext * ciphertext       # 逐轮自乘
            decrypted = ciphertext.decrypt()
            is_correct = decrypted == expected
            print(
                f"Round {round_idx:2d}: decrypted = {decrypted}, "
                f"matches expected = {is_correct}"
            )

            if not is_correct and failure_round is None:
                failure_round = round_idx
                print(
                    f">>> Distortion first observed at round {round_idx}: "
                    "decryption result no longer matches plaintext [1]."
                )
                break
        except Exception as exc:
            failure_round = round_idx
            print(f">>> Exception at round {round_idx}: {type(exc).__name__}: {exc}")
            break

    if failure_round is None:
        print("No exception or complete distortion observed within the configured rounds.")
    else:
        print(f"Recorded failure/distortion round: {failure_round}")
    print()


def experiment_ckks_repeated_self_multiplication(max_rounds=8):
    """
    CKKS 连续自乘实验：验证层级耗尽。

    CKKS 每次乘法后需要 rescale（消耗一个 40-bit 模数），
    当 moduli chain 耗尽时抛出 ValueError: scale out of bounds。

    设计: [1.0] 自乘避免数值爆炸，聚焦观察层级生命周期。
    预期: 约 2-3 轮后 scale 越界（coeff_mod_bit_sizes 仅 4 个 prime）。
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],     # 共 2 级乘法可用
    )
    context.global_scale = 2**40
    context.generate_relin_keys()

    ciphertext = ts.ckks_vector(context, [1.0])
    expected = [1.0]
    failure_round = None

    print("=== Part 3: CKKS Level Exhaustion Experiment ===")
    print("Plaintext seed: [1.0]")
    print("Expected plaintext after each self-multiplication: approximately [1.0]")

    for round_idx in range(1, max_rounds + 1):
        try:
            ciphertext = ciphertext * ciphertext     # 每次乘法消耗一级
            decrypted = ciphertext.decrypt()
            is_correct = almost_equal(decrypted, expected, tol=1e-2)
            print(
                f"Round {round_idx:2d}: decrypted = {decrypted}, "
                f"close to expected = {is_correct}"
            )
        except Exception as exc:
            failure_round = round_idx
            print(f">>> Exception at round {round_idx}: {type(exc).__name__}: {exc}")
            break

    if failure_round is None:
        print("No exception observed within the configured rounds.")
    else:
        print(f"Recorded exception round: {failure_round}")
    print()


def main():
    demo_ckks_basic()
    experiment_bfv_repeated_self_multiplication()
    experiment_ckks_repeated_self_multiplication()


if __name__ == "__main__":
    main()
