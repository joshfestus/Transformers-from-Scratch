"""Chapter 14, section 7. The same head in PyTorch, and the proof.

The point of this file is the two assertions at the bottom. The framework
is doing the arithmetic from ch14_attention.py and nothing else, and the
only way to believe that is to run both and compare the numbers.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ch14_attention import X, W_Q, W_K, W_V, attention

torch.set_printoptions(precision=3, sci_mode=False)


def attention_torch(x, w_q, w_k, w_v):
    """One head, written with PyTorch operations instead of NumPy."""
    queries = x @ w_q
    keys = x @ w_k
    values = x @ w_v
    d_k = keys.shape[-1]
    scores = queries @ keys.transpose(-2, -1) / d_k ** 0.5
    weights = torch.softmax(scores, dim=-1)
    return weights, weights @ values


def main():
    x = torch.tensor(X, dtype=torch.float64)
    w_q = torch.tensor(W_Q, dtype=torch.float64)
    w_k = torch.tensor(W_K, dtype=torch.float64)
    w_v = torch.tensor(W_V, dtype=torch.float64)

    weights, output = attention_torch(x, w_q, w_k, w_v)
    print("Attention weights from PyTorch")
    print(weights)
    print("\nOutput from PyTorch")
    print(output)

    # the same call, handed to the built-in fused implementation
    builtin = F.scaled_dot_product_attention(x @ w_q, x @ w_k, x @ w_v)

    _, _, _, _, _, np_weights, np_output = attention(X, W_Q, W_K, W_V)
    assert np.allclose(weights.numpy(), np_weights, atol=1e-12)
    assert np.allclose(output.numpy(), np_output, atol=1e-12)
    assert np.allclose(builtin.numpy(), np_output, atol=1e-12)
    print("\nNumPy, PyTorch and F.scaled_dot_product_attention agree")
    print("largest disagreement:",
          float(np.abs(builtin.numpy() - np_output).max()))


if __name__ == "__main__":
    main()
