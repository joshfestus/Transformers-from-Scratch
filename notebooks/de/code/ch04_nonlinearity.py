"""Kapitel 4. Warum ein Stapel linearer Schichten eine Schicht ist.

Zwei Matrixmultiplikationen hintereinander lassen sich immer durch eine
einzige Matrix ersetzen. Diese Datei beweist das mit Zahlen, statt es zu
behaupten, und misst dann, was die Aktivierungsfunktionen dagegen tun.
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
    """Die tanh-Näherung, die Implementierungen tatsächlich benutzen."""
    inner = np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)
    return 0.5 * z * (1.0 + np.tanh(inner))


def silu(z):
    return z / (1.0 + np.exp(-z))


def collapse_demo():
    """Zwei Schichten ohne Aktivierung, gegen eine gleichwertige."""
    two_steps = (X @ W1) @ W2
    combined = W1 @ W2
    one_step = X @ combined
    return two_steps, combined, one_step


def dead_units(seed=0, n=4000, d=64):
    """Wie viele ReLU-Einheiten eine Zufallseingabe abschaltet."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    w = rng.standard_normal((d, d)) / np.sqrt(d)
    for bias, label in [(0.0, "Bias 0.0"), (-1.0, "Bias -1.0")]:
        activated = relu(x @ w + bias)
        share = float((activated == 0.0).mean())
        yield label, share


def main():
    two_steps, combined, one_step = collapse_demo()
    print("ZWEI LINEARE SCHICHTEN, KEINE AKTIVIERUNG")
    print("  x durch W1, dann durch W2")
    print(two_steps)
    print("  die eine Matrix W1 @ W2, Form", combined.shape)
    print(combined)
    print("  x durch diese eine Matrix")
    print(one_step)
    print("  identisch:", np.allclose(two_steps, one_step))

    print("\nDIESELBEN ZWEI SCHICHTEN MIT EINER RELU DAZWISCHEN")
    bent = relu(X @ W1) @ W2
    print(bent)
    print("  keine einzelne Matrix bildet das nach, und darum geht es")

    print("\nDREI AKTIVIERUNGSFUNKTIONEN AN FÜNF EINGABEN")
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    print("  Eingabe", z)
    print("  relu   ", relu(z))
    print("  gelu   ", gelu(z))
    print("  silu   ", silu(z))

    print("\nWIE VIELE RELU-EINHEITEN BEI NULL LIEGEN")
    for label, share in dead_units():
        print(f"  {label:<12} {100 * share:.1f} Prozent abgeschaltet")


if __name__ == "__main__":
    main()
