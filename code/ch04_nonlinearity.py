"""Chapter 4. Why a stack of linear layers is one linear layer.

Two matrix multiplications in a row can always be replaced by a single
matrix. This file proves that with numbers rather than asserting it, and
then measures what the activation functions do about it.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

W1 = np.array([[0.5, -1.0, 2.0],
               [1.5, 0.0, -0.5]])
W2 = np.array([[1.0, 0.0],
               [2.0, -1.0],
               [0.0, 1.0]])
X = np.array([[1.0, 2.0],
              [3.0, -1.0]])


def relu(z):
    return np.maximum(z, 0.0)


def gelu(z):
    """The tanh approximation, which is what implementations use."""
    inner = np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)
    return 0.5 * z * (1.0 + np.tanh(inner))


def silu(z):
    return z / (1.0 + np.exp(-z))


def collapse_demo():
    """Two layers with no activation, against one equivalent layer."""
    two_steps = (X @ W1) @ W2
    combined = W1 @ W2
    one_step = X @ combined
    return two_steps, combined, one_step


def dead_units(seed=0, n=4000, d=64):
    """How many ReLU units are switched off by a random input."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    w = rng.standard_normal((d, d)) / np.sqrt(d)
    for bias, label in [(0.0, "bias 0.0"), (-1.0, "bias -1.0")]:
        activated = relu(x @ w + bias)
        share = float((activated == 0.0).mean())
        yield label, share


def main():
    two_steps, combined, one_step = collapse_demo()
    print("TWO LINEAR LAYERS, NO ACTIVATION")
    print("  x through W1 then W2")
    print(two_steps)
    print("  the single matrix W1 @ W2, shape", combined.shape)
    print(combined)
    print("  x through that one matrix")
    print(one_step)
    print("  identical:", np.allclose(two_steps, one_step))

    print("\nTHE SAME TWO LAYERS WITH A RELU BETWEEN THEM")
    bent = relu(X @ W1) @ W2
    print(bent)
    print("  no single matrix can reproduce this, and that is the point")

    print("\nTHREE ACTIVATION FUNCTIONS AT FIVE INPUTS")
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    print("  input", z)
    print("  relu ", relu(z))
    print("  gelu ", gelu(z))
    print("  silu ", silu(z))

    print("\nHOW MANY RELU UNITS SIT AT NOUGHT")
    for label, share in dead_units():
        print(f"  {label:<12} {100 * share:.1f} per cent switched off")


if __name__ == "__main__":
    main()
