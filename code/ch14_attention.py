"""Chapter 14. One attention head, computed with numbers you can check.

Four tokens, a model dimension of four, one head, a key dimension of four.
Every weight here is a nought or a one and every embedding is a small
whole number, so every product below can be checked with a pencil. The
scores divide by the square root of four, which is exactly two.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

TOKENS = ["the", "cat", "drank", "milk"]

# One row per token. In a trained model these are learned. Here they are
# chosen so that "cat" and "milk" share a feature and "drank" does not.
X = np.array([
    [1.0, 0.0, 1.0, 0.0],   # the
    [0.0, 2.0, 0.0, 1.0],   # cat
    [1.0, 0.0, 2.0, 0.0],   # drank
    [0.0, 1.0, 0.0, 2.0],   # milk
])

W_Q = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
])

W_K = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

W_V = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 1.0],
])


def softmax(scores):
    """Row-wise softmax, shifted by the row max so it cannot overflow."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def attention(x, w_q, w_k, w_v):
    """One head of scaled dot-product attention, keeping every step."""
    queries = x @ w_q
    keys = x @ w_k
    values = x @ w_v
    d_k = keys.shape[-1]
    raw = queries @ keys.T
    scaled = raw / np.sqrt(d_k)
    weights = softmax(scaled)
    output = weights @ values
    return queries, keys, values, raw, scaled, weights, output


def show(name, matrix, cols):
    """Print one labelled matrix the way the book prints it."""
    print(f"\n{name}   shape {matrix.shape}")
    print(" " * 7 + "".join(f"{c:>8}" for c in cols))
    for label, row in zip(TOKENS, matrix):
        print(f"{label:>6} " + "".join(f"{v:>8.3f}" for v in row))


def main():
    q, k, v, raw, scaled, weights, out = attention(X, W_Q, W_K, W_V)
    show("X, the input embeddings", X, ["e0", "e1", "e2", "e3"])
    show("Q = X W_Q", q, ["q0", "q1", "q2", "q3"])
    show("K = X W_K", k, ["k0", "k1", "k2", "k3"])
    show("V = X W_V", v, ["v0", "v1", "v2", "v3"])
    show("Raw scores, Q K-transpose", raw, TOKENS)
    show("Scaled scores, every entry divided by 2", scaled, TOKENS)
    show("Attention weights, softmax along each row", weights, TOKENS)
    show("Output, weights times V", out, ["o0", "o1", "o2", "o3"])
    print("\nRow sums of the attention weights:", weights.sum(axis=1))


if __name__ == "__main__":
    main()
