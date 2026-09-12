"""Chapter 10. Position, the information attention throws away.

Attention treats its input as a set. Shuffle the rows and the same rows
come out, shuffled the same way, which means order carries no
information at all until something puts it there. This file shows that
failure first, then the two usual repairs.
"""
import numpy as np

from ch14_attention import W_K, W_Q, W_V, X, attention

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def positional_encoding(n, d_model):
    """The sinusoidal ladder. Even columns are sines, odd ones cosines."""
    position = np.arange(n)[:, None]
    index = np.arange(0, d_model, 2)[None, :]
    rate = np.exp(-np.log(10000.0) * index / d_model)
    encoding = np.zeros((n, d_model))
    encoding[:, 0::2] = np.sin(position * rate)
    encoding[:, 1::2] = np.cos(position * rate)
    return encoding


def rotate(vector, angle):
    """Rotary position embedding, on a single pair of dimensions."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vector[0] - s * vector[1],
                     s * vector[0] + c * vector[1]])


def order_blindness():
    """Run attention on a sentence and on a shuffled copy of it."""
    order = np.array([3, 0, 2, 1])
    plain = attention(X, W_Q, W_K, W_V)[-1]
    shuffled = attention(X[order], W_Q, W_K, W_V)[-1]
    return order, plain, shuffled


def main():
    order, plain, shuffled = order_blindness()
    print("ATTENTION ON THE SENTENCE AS WRITTEN")
    print(plain)
    print("\nTHE SAME FOUR ROWS, SHUFFLED, THROUGH THE SAME ATTENTION")
    print(shuffled)
    print("\nTHE SECOND IS THE FIRST WITH ITS ROWS IN THE NEW ORDER")
    print("  identical:", np.allclose(shuffled, plain[order]))
    print("  so nothing inside attention knows which word came first")

    print("\nTHE SINUSOIDAL ENCODING, FOUR POSITIONS BY SIX NUMBERS")
    encoding = positional_encoding(4, 6)
    for i, row in enumerate(encoding):
        print(f"  position {i}  {row}")

    print("\nEACH COLUMN PAIR TURNS AT ITS OWN RATE")
    d_model = 64
    for index in [0, 2, 8, 32]:
        rate = np.exp(-np.log(10000.0) * index / d_model)
        period = 2 * np.pi / rate
        print(f"  columns {index} and {index + 1}: one full turn every "
              f"{period:,.0f} positions")

    print("\nTHE DOT PRODUCT OF TWO ENCODINGS DEPENDS ON THE GAP")
    long_encoding = positional_encoding(40, 64)
    print("  gap   dot product with position 10")
    for gap in [0, 1, 2, 5, 10, 20]:
        value = float(long_encoding[10] @ long_encoding[10 + gap])
        print(f"  {gap:<5} {value:.3f}")

    print("\nROTARY POSITIONS ROTATE INSTEAD OF ADDING")
    vector = np.array([1.0, 0.0])
    for position in range(4):
        turned = rotate(vector, position * np.pi / 6)
        print(f"  position {position}  {turned}")
    a = rotate(vector, 1 * np.pi / 6)
    b = rotate(vector, 3 * np.pi / 6)
    c = rotate(vector, 5 * np.pi / 6)
    d = rotate(vector, 7 * np.pi / 6)
    print(f"  dot of positions 1 and 3  {float(a @ b):.3f}")
    print(f"  dot of positions 5 and 7  {float(c @ d):.3f}")
    print("  the same gap gives the same answer, wherever it happens")


if __name__ == "__main__":
    main()
