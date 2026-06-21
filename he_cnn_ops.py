"""Encrypted CNN operations using TenSEAL CKKS.

Supports multi-channel (RGB) input.  Avoids pack_vectors by computing
Flatten+FC1 as: sum over channels of matmul(ch_i, W_i.T) + bias.
"""

from typing import List
import numpy as np
import tenseal as ts


def im2col_encode_channel(context, channel_2d, kernel_rows, kernel_cols,
                         stride=1, padding=0):
    if padding > 0:
        channel_2d = np.pad(channel_2d, padding, mode="constant", constant_values=0.0)
    enc, windows_nb = ts.im2col_encoding(
        context, channel_2d, kernel_rows, kernel_cols, stride)
    return enc, windows_nb


def encrypted_conv2d(enc_channels, windows_nb, weight, bias=None):
    """Multi-channel encrypted Conv2d.

    weight: [OC, IC, KH, KW]
    Returns: list of OC CKKSVectors
    """
    OC, IC, KH, KW = weight.shape
    assert len(enc_channels) == IC
    results = []
    for oc in range(OC):
        acc = None
        for ic in range(IC):
            kernel_flat = weight[oc, ic].reshape(1, KH * KW).astype(np.float64).tolist()
            contrib = enc_channels[ic].conv2d_im2col(kernel_flat, windows_nb)
            acc = contrib if acc is None else acc + contrib
        if bias is not None:
            acc += float(bias[oc])
        results.append(acc)
    return results


def encrypted_square(enc):
    return enc.square()


def encrypted_flatten_and_fc(channel_vecs, fc1_weight, fc1_bias=None):
    """Flatten + FC1: sum_i matmul(ch_i, W_i.T) + bias.

    fc1_weight: [out_features, total_in_features]
    """
    out_ch = fc1_weight.shape[0]
    per_ch = channel_vecs[0].size()
    total_in = len(channel_vecs) * per_ch
    assert fc1_weight.shape[1] == total_in, f"{fc1_weight.shape[1]} != {total_in}"

    result = None
    for i, vec in enumerate(channel_vecs):
        w_slice = fc1_weight[:, i * per_ch: (i + 1) * per_ch]
        contrib = vec.matmul(w_slice.T.tolist())
        result = contrib if result is None else result + contrib
    if fc1_bias is not None:
        result += fc1_bias.tolist()
    return result


def encrypted_linear(enc, weight, bias):
    out = enc.matmul(weight.T.tolist())
    if bias is not None:
        out += bias.tolist()
    return out


def encrypted_cnn_inference(enc_channels, windows_nb,
                           conv_weight, conv_bias,
                           fc1_weight, fc1_bias,
                           fc2_weight, fc2_bias):
    """Full pipeline: Conv → Square → Flatten+FC1 → Square → FC2."""
    after_conv = encrypted_conv2d(enc_channels, windows_nb, conv_weight, conv_bias)
    for i in range(len(after_conv)):
        after_conv[i] = encrypted_square(after_conv[i])

    enc_fc1 = encrypted_flatten_and_fc(after_conv, fc1_weight, fc1_bias)
    enc_fc1 = encrypted_square(enc_fc1)
    return encrypted_linear(enc_fc1, fc2_weight, fc2_bias)
