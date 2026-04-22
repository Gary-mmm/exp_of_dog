# 实验结果记录

## 预备实验
'''
=== Part 1: Basic CKKS Demo ===
Plaintext X: [1.5, 2.5]
Plaintext Y: [3.4, 4.5]
Decrypted Enc(X) + Enc(Y): [4.900000000164191, 6.999999997831435]
Plaintext X + Y:           [4.9, 7.0]
Decrypted Enc(X) * Enc(Y): [5.1000006852354565, 11.250001500839366]
Plaintext X * Y:           [5.1, 11.25]
Addition check passed: True
Multiplication check passed: True

=== Part 2: BFV Noise Budget Experiment ===
Plaintext seed: [1]
Expected plaintext after each self-multiplication: [1]
Round  1: decrypted = [1], matches expected = True
Round  2: decrypted = [1], matches expected = True
Round  3: decrypted = [1], matches expected = True
Round  4: decrypted = [1], matches expected = True
Round  5: decrypted = [265689], matches expected = False
>>> Distortion first observed at round 5: decryption result no longer matches plaintext [1].
Recorded failure/distortion round: 5

=== Part 3: CKKS Level Exhaustion Experiment ===
Plaintext seed: [1.0]
Expected plaintext after each self-multiplication: approximately [1.0]
Round  1: decrypted = [1.0000001339856945], close to expected = True
Round  2: decrypted = [1.0000009384452744], close to expected = True
>>> Exception at round 3: ValueError: scale out of bounds
Recorded exception round: 3
'''
