"""Kapitel 6. Ein ganzer Rückwärtsdurchlauf, von Hand und maschinell.

Zwei Eingaben, zwei verdeckte Einheiten mit ReLU, eine Ausgabe und ein
quadratischer Fehler gegen ein einzelnes Ziel. Jedes Gewicht ist eine
ganze Zahl oder ein Halb, also lässt sich jede Ableitung unten mit dem
Bleistift prüfen, und jede wird danach mit einer finiten Differenz
noch einmal geprüft.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

X = np.array([1.0, 2.0])            # ein Beispiel, zwei Merkmale
Y = 1.0                             # das Ziel

W1 = np.array([[0.5, -0.5],         # 2 mal 2, eine Spalte pro Einheit
               [1.0, 0.5]])
B1 = np.array([0.0, 0.5])
W2 = np.array([1.0, -1.0])          # 2 mal 1, flach geschrieben
B2 = 0.5


def relu(z):
    return np.maximum(z, 0.0)


def forward(x, w1, b1, w2, b2, y):
    """Jeder Zwischenwert, weil der Rückwärtsdurchlauf ihn braucht."""
    h_pre = x @ w1 + b1
    h = relu(h_pre)
    pred = h @ w2 + b2
    loss = 0.5 * (pred - y) ** 2
    return h_pre, h, pred, loss


def backward(x, w1, b1, w2, b2, y):
    """Die Kettenregel, einmal pro Schicht, vom Loss rückwärts."""
    h_pre, h, pred, _ = forward(x, w1, b1, w2, b2, y)
    d_pred = pred - y                       # aus 0.5 (pred - y) hoch 2
    d_w2 = h * d_pred                       # pred = h . w2 + b2
    d_b2 = d_pred
    d_h = w2 * d_pred
    d_h_pre = d_h * (h_pre > 0)             # ReLU lässt durch oder sperrt
    d_w1 = np.outer(x, d_h_pre)             # h_pre = x @ w1 + b1
    d_b1 = d_h_pre
    return d_w1, d_b1, d_w2, d_b2


def loss_only(w1, b1, w2, b2):
    return forward(X, w1, b1, w2, b2, Y)[3]


def numeric_gradient(eps=1e-6):
    """Dieselben Gradienten, gemessen durch Anstupsen jedes Parameters."""
    g_w1 = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            up, down = W1.copy(), W1.copy()
            up[i, j] += eps
            down[i, j] -= eps
            g_w1[i, j] = (loss_only(up, B1, W2, B2)
                          - loss_only(down, B1, W2, B2)) / (2 * eps)
    g_w2 = np.zeros_like(W2)
    for j in range(W2.shape[0]):
        up, down = W2.copy(), W2.copy()
        up[j] += eps
        down[j] -= eps
        g_w2[j] = (loss_only(W1, B1, up, B2)
                   - loss_only(W1, B1, down, B2)) / (2 * eps)
    return g_w1, g_w2


def sweep_squared_error():
    """Fehler der Gradientenprüfung je Schrittweite, für das Netz oben."""
    exact_w1, _, exact_w2, _ = backward(X, W1, B1, W2, B2, Y)
    rows = []
    for power in range(-11, 0):
        eps = 10.0 ** power
        g_w1, g_w2 = numeric_gradient(eps)
        err = max(np.abs(g_w1 - exact_w1).max(),
                  np.abs(g_w2 - exact_w2).max())
        rows.append((eps, float(err)))
    return rows


def logit_loss(w):
    """Kreuzentropie auf drei Logits, als Funktion des ersten."""
    return -np.log(softmax(np.array([w, 1.0, 0.0]))[0])


def sweep_cross_entropy(w=2.0):
    """Derselbe Durchlauf bei einem Loss, der nicht quadratisch ist."""
    exact = softmax(np.array([w, 1.0, 0.0]))[0] - 1.0
    rows = []
    for power in range(-12, 0):
        eps = 10.0 ** power
        measured = (logit_loss(w + eps) - logit_loss(w - eps)) / (2 * eps)
        rows.append((eps, float(abs(measured - exact))))
    return rows


def softmax(z):
    shifted = z - z.max()
    e = np.exp(shifted)
    return e / e.sum()


def cross_entropy_gradient(logits, correct):
    """Das Ergebnis, auf das sich der Rest des Buches stützt: Der Gradient
    von Softmax mit Kreuzentropie ist p minus One-Hot."""
    p = softmax(logits)
    target = np.zeros_like(p)
    target[correct] = 1.0
    return p, p - target


def main():
    h_pre, h, pred, loss = forward(X, W1, B1, W2, B2, Y)
    print("VORWÄRTS")
    print("  h_pre  ", h_pre)
    print("  h      ", h)
    print("  pred   ", pred)
    print("  loss   ", loss)

    d_w1, d_b1, d_w2, d_b2 = backward(X, W1, B1, W2, B2, Y)
    print("\nRÜCKWÄRTS")
    print("  dL/dpred", pred - Y)
    print("  dL/dW2  ", d_w2)
    print("  dL/db2  ", d_b2)
    print("  dL/dW1  ", d_w1[0], "und", d_w1[1])
    print("  dL/db1  ", d_b1)

    g_w1, g_w2 = numeric_gradient()
    print("\nGEPRÜFT GEGEN EINE FINITE DIFFERENZ, eps = 1e-6")
    print("  numerisch dL/dW1", g_w1[0], "und", g_w1[1])
    print("  numerisch dL/dW2", g_w2)
    worst = max(np.abs(g_w1 - d_w1).max(), np.abs(g_w2 - d_w2).max())
    print("  größte Abweichung", f"{worst:.2e}")

    print("\nSCHRITTWEITE GEGEN FEHLER, zwei Losses")
    print("  eps        quadr. Fehler   Kreuzentropie")
    for (eps, a), (_, b) in zip(sweep_squared_error()[1:],
                                sweep_cross_entropy()[2:]):
        print(f"  {eps:.0e}   {a:12.2e}   {b:13.2e}")

    print("\nSOFTMAX MIT KREUZENTROPIE, Gradient ist p minus One-Hot")
    p, g = cross_entropy_gradient(np.array([2.0, 1.0, 0.0]), correct=0)
    print("  Wahrscheinlichkeiten", p)
    print("  Gradient            ", g)


if __name__ == "__main__":
    main()
