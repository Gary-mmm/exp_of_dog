import math

import tenseal as ts


def almost_equal(vec1, vec2, tol=1e-3):
    return all(math.isclose(a, b, rel_tol=tol, abs_tol=tol) for a, b in zip(vec1, vec2))


def demo_ckks_basic():
    """基础 CKKS 演示：加密、同态加法、同态乘法、解密校验。"""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()

    x = [1.5, 2.5]
    y = [3.4, 4.5]

    enc_x = ts.ckks_vector(context, x)
    enc_y = ts.ckks_vector(context, y)

    enc_add = enc_x + enc_y
    enc_mul = enc_x * enc_y

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
    BFV 连续自乘实验：
    选择明文 [1]，理论上无论平方多少次都应保持为 [1]。
    因此，一旦解密结果偏离 [1]，可视为噪声预算耗尽导致的失真。
    """
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=8192,
        plain_modulus=1032193,
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
            ciphertext = ciphertext * ciphertext
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
    CKKS 连续自乘实验：
    用 [1.0] 避免明文数值爆炸，重点观察 scale/层级耗尽导致的异常。
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
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
            ciphertext = ciphertext * ciphertext
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
