"""Chapter 17. Cross-attention: queries from here, keys from there.

Self-attention compares a sequence against itself. Cross-attention
compares one sequence against another, which is how a translation model
lets the sentence it is writing consult the sentence it was given. The
arithmetic is identical. Only where the three projections come from
changes.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

rng = np.random.default_rng(0)

SOURCE = ["le", "chat", "a", "bu", "le", "lait"]
TARGET = ["the", "cat", "drank"]
D_MODEL, D_HEAD = 8, 4


def softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def attend(queries_from, keys_from, w_q, w_k, w_v):
    """One head. The two sources are the only thing that changes."""
    q = queries_from @ w_q
    k = keys_from @ w_k
    v = keys_from @ w_v
    scores = q @ k.T / np.sqrt(k.shape[-1])
    weights = softmax(scores)
    return weights, weights @ v


def main():
    source = rng.standard_normal((len(SOURCE), D_MODEL))
    target = rng.standard_normal((len(TARGET), D_MODEL))
    w_q = rng.standard_normal((D_MODEL, D_HEAD)) / np.sqrt(D_MODEL)
    w_k = rng.standard_normal((D_MODEL, D_HEAD)) / np.sqrt(D_MODEL)
    w_v = rng.standard_normal((D_MODEL, D_HEAD)) / np.sqrt(D_MODEL)

    print("TWO SEQUENCES OF DIFFERENT LENGTHS")
    print(f"  source, {len(SOURCE)} tokens  {SOURCE}")
    print(f"  target, {len(TARGET)} tokens  {TARGET}")
    print(f"  source shape {source.shape}   target shape {target.shape}")

    print("\nSELF-ATTENTION ON THE TARGET")
    weights, out = attend(target, target, w_q, w_k, w_v)
    print(f"  weights {weights.shape}   output {out.shape}")
    print("  every target position against every target position")

    print("\nCROSS-ATTENTION, TARGET QUERIES AGAINST SOURCE KEYS")
    weights, out = attend(target, source, w_q, w_k, w_v)
    print(f"  weights {weights.shape}   output {out.shape}")
    print("  one row per target position, one column per source position")

    print("\nTHE WEIGHTS, ROW BY ROW")
    header = "        " + "".join(f"{w:>8}" for w in SOURCE)
    print(header)
    for token, row in zip(TARGET, weights):
        print(f"  {token:>6}" + "".join(f"{v:>8.3f}" for v in row))
    print("  row sums", weights.sum(axis=1))

    print("\nWHAT CHANGES AND WHAT DOES NOT")
    print("  the output has one row per query, always")
    print("  so cross-attention returns a target-shaped answer built")
    print("  entirely out of source values")

    print("\nTHE OUTPUT LENGTH FOLLOWS THE QUERIES, NOT THE KEYS")
    for n_target in [1, 3, 9]:
        t = rng.standard_normal((n_target, D_MODEL))
        _, o = attend(t, source, w_q, w_k, w_v)
        print(f"  {n_target} queries against {len(SOURCE)} keys "
              f"gives {o.shape}")


if __name__ == "__main__":
    main()
