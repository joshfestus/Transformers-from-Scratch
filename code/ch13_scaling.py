"""Chapter 13. Why the scores are divided by the square root of d_k.

The division is usually stated and not justified. It has a reason, the
reason is arithmetic, and this file measures it. Two random vectors of
length d, each component drawn independently with mean nought and
variance one, have a dot product whose variance is d. So the scores grow
with the key dimension, and a softmax fed growing numbers stops being a
soft anything.
"""
import numpy as np

rng = np.random.default_rng(0)

DIMS = [4, 16, 64, 256, 1024]
TRIALS = 20000


def softmax(scores):
    """Row-wise softmax, shifted by the row max so it cannot overflow."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def dot_variance(d, trials=TRIALS):
    """Measure the variance of q . k for random q and k of length d."""
    q = rng.standard_normal((trials, d))
    k = rng.standard_normal((trials, d))
    dots = (q * k).sum(axis=1)
    return dots.var()


def peak_weight(d, n_keys=8, trials=4000, scale=True):
    """How much of the attention lands on the single winning key."""
    q = rng.standard_normal((trials, d))
    k = rng.standard_normal((trials, n_keys, d))
    scores = np.einsum("td,tkd->tk", q, k)
    if scale:
        scores = scores / np.sqrt(d)
    return softmax(scores).max(axis=1).mean()


def main():
    print("d_k    measured variance   d_k itself   ratio")
    for d in DIMS:
        v = dot_variance(d)
        print(f"{d:>4}   {v:>16.1f}   {d:>10}   {v / d:>5.3f}")

    print("\nlargest attention weight of eight, averaged over 4000 draws")
    print("d_k    unscaled   scaled by sqrt(d_k)")
    for d in DIMS:
        print(f"{d:>4}   {peak_weight(d, scale=False):>8.3f}   "
              f"{peak_weight(d, scale=True):>19.3f}")

    print("\none unscaled row of scores at d_k = 256, and its weights")
    d = 256
    q = rng.standard_normal(d)
    k = rng.standard_normal((4, d))
    raw = k @ q
    def show(label, values, places):
        print(label, np.array2string(values, precision=places,
                                     floatmode="fixed"))

    show("raw     ", raw, 1)
    show("weights ", softmax(raw), 3)
    show("scaled  ", raw / np.sqrt(d), 1)
    show("weights ", softmax(raw / np.sqrt(d)), 3)


if __name__ == "__main__":
    main()
