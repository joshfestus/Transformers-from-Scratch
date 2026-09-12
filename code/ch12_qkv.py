"""Chapter 12. Query, key and value, as a lookup that is allowed to blend.

An ordinary dictionary matches a key exactly or fails. Attention scores
every key, turns those scores into amounts, and returns a mixture. This
file puts the two side by side.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

KEYS = np.array([[1.0, 0.0],
                 [0.0, 1.0],
                 [-1.0, 0.0],
                 [0.0, -1.0]])
VALUES = np.array([[10.0, 0.0],
                   [0.0, 10.0],
                   [-10.0, 0.0],
                   [0.0, -10.0]])
NAMES = ["north", "east", "south", "west"]


def softmax(scores):
    shifted = scores - scores.max()
    e = np.exp(shifted)
    return e / e.sum()


def exact_lookup(query):
    """A dictionary. A key matches exactly, or nothing comes back."""
    for name, key, value in zip(NAMES, KEYS, VALUES):
        if np.allclose(query, key):
            return name, value
    return None, None


def soft_lookup(query, sharpness=4.0):
    """Score every key, normalise, then blend the values by those."""
    scores = KEYS @ query * sharpness
    weights = softmax(scores)
    return weights, weights @ VALUES


def projections(x, w_q, w_k, w_v):
    """One input row becomes three different rows."""
    return x @ w_q, x @ w_k, x @ w_v


def main():
    print("AN ORDINARY LOOKUP")
    for query in [np.array([1.0, 0.0]), np.array([0.7, 0.7])]:
        name, value = exact_lookup(query)
        found = f"{name}, {value}" if name else "nothing at all"
        print(f"  query {query}  ->  {found}")

    print("\nTHE SAME TWO QUERIES, SCORED AND BLENDED")
    for query in [np.array([1.0, 0.0]), np.array([0.7, 0.7])]:
        weights, blended = soft_lookup(query)
        print(f"  query {query}")
        for name, weight in zip(NAMES, weights):
            print(f"    {name:<7} {weight:.3f}")
        print(f"    result {blended}")

    print("\nA QUERY THAT MATCHES NOTHING STILL GETS AN ANSWER")
    weights, blended = soft_lookup(np.array([0.0, 0.0]))
    print("  query [0. 0.]")
    print("    weights", weights)
    print("    result ", blended)
    print("  every key scored nought, so the answer is an even blend")

    print("\nONE INPUT, THREE DIFFERENT VECTORS")
    x = np.array([1.0, 2.0])
    w_q = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_k = np.array([[0.0, 1.0], [1.0, 0.0]])
    w_v = np.array([[2.0, 0.0], [0.0, 0.5]])
    q, k, v = projections(x, w_q, w_k, w_v)
    print("  input   ", x)
    print("  query   ", q, " what this position is looking for")
    print("  key     ", k, " what it offers to be matched against")
    print("  value   ", v, " what it contributes when attended to")
    print("  three roles, three matrices, one input")


if __name__ == "__main__":
    main()
