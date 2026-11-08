"""Kapitel 14, Abschnitt 7. Derselbe Head in PyTorch, und der Beweis.

Worauf es in dieser Datei ankommt, sind die Assertions am Ende. Das
Framework rechnet genau das, was ch14_attention.py rechnet, und nichts
sonst, und glauben kann man das nur, wenn man beides laufen lässt und
die Zahlen vergleicht.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ch14_attention import X, W_Q, W_K, W_V, attention

torch.set_printoptions(precision=3, sci_mode=False)


def attention_torch(x, w_q, w_k, w_v):
    """Ein Head, mit PyTorch-Operationen statt mit NumPy geschrieben."""
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
    print("Attention-Gewichte aus PyTorch")
    print(weights)
    print("\nAusgabe aus PyTorch")
    print(output)

    # derselbe Aufruf, übergeben an die eingebaute, fusionierte Version
    builtin = F.scaled_dot_product_attention(x @ w_q, x @ w_k, x @ w_v)

    _, _, _, _, _, np_weights, np_output = attention(X, W_Q, W_K, W_V)
    assert np.allclose(weights.numpy(), np_weights, atol=1e-12)
    assert np.allclose(output.numpy(), np_output, atol=1e-12)
    assert np.allclose(builtin.numpy(), np_output, atol=1e-12)
    print("\nNumPy, PyTorch und F.scaled_dot_product_attention"
          " stimmen überein")
    print("größte Abweichung:",
          float(np.abs(builtin.numpy() - np_output).max()))


if __name__ == "__main__":
    main()
