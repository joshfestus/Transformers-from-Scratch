"""Chapter 16. Stopping a position from seeing what it should not.

Two masks. A causal mask keeps a position from reading the future, which
is what makes generation possible. A padding mask keeps real positions
from attending to the filler added to square off a batch. Both are
applied before the softmax, and this file shows what happens when they
are applied after it instead.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

SCORES = np.array([[2.0, 1.0, 0.5, 0.0],
                   [1.0, 3.0, 1.5, 0.5],
                   [0.5, 1.0, 2.0, 1.0],
                   [0.0, 0.5, 1.5, 2.5]])


def softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(n):
    """True where a query at row i may read a key at column j."""
    return np.tril(np.ones((n, n), dtype=bool))


def padding_mask(lengths, n):
    """True for real positions, False for the filler after them."""
    return np.arange(n)[None, :] < np.asarray(lengths)[:, None]


def masked_before(scores, allowed):
    """The right way. Push forbidden scores to minus infinity."""
    return softmax(np.where(allowed, scores, -np.inf))


def masked_after(scores, allowed):
    """The wrong way. Normalise first, then zero what is forbidden."""
    return np.where(allowed, softmax(scores), 0.0)


def main():
    n = 4
    allowed = causal_mask(n)
    print("THE CAUSAL MASK, TRUE MEANS ALLOWED")
    for row in allowed:
        print("  " + "  ".join("yes" if v else " . " for v in row))

    print("\nSCORES BEFORE ANY MASK")
    print(SCORES)

    print("\nMASKED BEFORE THE SOFTMAX, WHICH IS CORRECT")
    right = masked_before(SCORES, allowed)
    print(right)
    print("  row sums", right.sum(axis=1))

    print("\nMASKED AFTER THE SOFTMAX, WHICH IS NOT")
    wrong = masked_after(SCORES, allowed)
    print(wrong)
    sums = wrong.sum(axis=1)
    print("  row sums", sums)
    print(f"  the first row keeps only {sums[0]:.3f} of its weight,")
    print("  and the rest of it has gone nowhere at all")

    print("\nHOW FAR APART THE TWO ARE")
    print(f"  largest difference {np.abs(right - wrong).max():.3f}")

    print("\nA PADDING MASK FOR A BATCH OF THREE")
    lengths = [4, 2, 3]
    pad = padding_mask(lengths, n)
    for length, row in zip(lengths, pad):
        marks = "  ".join("yes" if v else " . " for v in row)
        print(f"  length {length}   {marks}")

    print("\nA MASK CONSTANT THAT IS NOT NEGATIVE ENOUGH")
    for constant in [-6.0, -20.0, -1e4]:
        leaked = softmax(np.where(allowed, SCORES, constant))[0, 1:].sum()
        print(f"  masking with {constant:>9}  forbidden weight left "
              f"{leaked:.3e}")
    print("  a modest constant leaks. A large one underflows in")
    print("  double precision, and would not in half precision. Minus")
    print("  infinity is exact whatever the number format")

    print("\nAND THE CASE MINUS INFINITY GETS WRONG")
    nothing = np.zeros((1, 4), dtype=bool)
    with np.errstate(invalid="ignore"):
        blocked = masked_before(SCORES[:1], nothing)
    print("  a row where every position is masked, using -inf")
    print("  ", blocked)
    print("  every score is -inf, so the sum is nought and the division")
    print("  is undefined. A fully padded row has to be special-cased,")
    print("  which is why library implementations carry that check")


if __name__ == "__main__":
    main()
