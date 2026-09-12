"""Chapter 5. Turning scores into probabilities, without overflowing.

Softmax appears three times in a transformer: inside every attention
head, at the output head, and in the loss. This file is about the one
function and the one trick that keeps it from breaking.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


def naive_softmax(scores):
    """The definition, exactly as written. It overflows."""
    e = np.exp(scores)
    return e / e.sum()


def softmax(scores, temperature=1.0):
    """The same function, shifted by the row maximum first."""
    scaled = np.asarray(scores, dtype=float) / temperature
    shifted = scaled - scaled.max()
    e = np.exp(shifted)
    return e / e.sum()


def temperature_table(scores, temperatures):
    return [(t, softmax(scores, t)) for t in temperatures]


def main():
    scores = np.array([2.0, 1.0, 0.1, -1.0])
    print("FOUR SCORES BECOME FOUR PROBABILITIES")
    print("  scores       ", scores)
    print("  probabilities", softmax(scores))
    print("  they add to  ", softmax(scores).sum())

    print("\nTHE ORDER NEVER CHANGES, ONLY THE SPACING")
    for t, p in temperature_table(scores, [0.25, 0.5, 1.0, 2.0, 8.0]):
        print(f"  temperature {t:<5} {p}  top {p.max():.3f}")

    print("\nWHY THE SHIFT IS NEEDED")
    big = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = naive_softmax(big)
    print("  scores        ", big)
    print("  without shift ", naive)
    print("  with shift    ", softmax(big))
    print("  the shift changes nothing mathematically, because")
    print("  every exponential is divided by the same constant")

    print("\nAND THE OTHER END OF THE RANGE")
    small = np.array([-1000.0, -1001.0, -1002.0])
    with np.errstate(under="ignore", invalid="ignore"):
        naive = naive_softmax(small)
    print("  scores        ", small)
    print("  without shift ", naive)
    print("  with shift    ", softmax(small))


if __name__ == "__main__":
    main()
