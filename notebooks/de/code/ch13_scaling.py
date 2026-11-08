"""Kapitel 13. Warum die Scores durch die Wurzel aus d_k geteilt werden.

Die Division wird meistens genannt und nicht begründet. Sie hat einen
Grund, der Grund ist Arithmetik, und diese Datei misst ihn. Zwei
zufällige Vektoren der Länge d, jede Komponente unabhängig gezogen mit
Mittelwert null und Varianz eins, haben ein Skalarprodukt mit der
Varianz d. Die Scores wachsen also mit der Key-Dimension, und eine
Softmax, die immer größere Zahlen bekommt, ist bald nicht mehr weich.
"""
import numpy as np

rng = np.random.default_rng(0)

DIMS = [4, 16, 64, 256, 1024]
TRIALS = 20000


def softmax(scores):
    """Softmax je Zeile, ums Zeilenmaximum verschoben: kein Überlauf."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def dot_variance(d, trials=TRIALS):
    """Die Varianz von q . k für zufällige q und k der Länge d messen."""
    q = rng.standard_normal((trials, d))
    k = rng.standard_normal((trials, d))
    dots = (q * k).sum(axis=1)
    return dots.var()


def peak_weight(d, n_keys=8, trials=4000, scale=True):
    """Wie viel der Attention auf dem einen siegreichen Key landet."""
    q = rng.standard_normal((trials, d))
    k = rng.standard_normal((trials, n_keys, d))
    scores = np.einsum("td,tkd->tk", q, k)
    if scale:
        scores = scores / np.sqrt(d)
    return softmax(scores).max(axis=1).mean()


def main():
    print("d_k    gemessene Varianz   d_k selbst   Verhältnis")
    for d in DIMS:
        v = dot_variance(d)
        print(f"{d:>4}   {v:>16.1f}   {d:>10}   {v / d:>10.3f}")

    print("\ngrößtes von acht Attention-Gewichten, Mittel aus 4000 Zügen")
    print("d_k    unskaliert   skaliert mit sqrt(d_k)")
    for d in DIMS:
        print(f"{d:>4}   {peak_weight(d, scale=False):>10.3f}   "
              f"{peak_weight(d, scale=True):>22.3f}")

    print("\neine unskalierte Zeile Scores bei d_k = 256, mit Gewichten")
    d = 256
    q = rng.standard_normal(d)
    k = rng.standard_normal((4, d))
    raw = k @ q
    def show(label, values, places):
        print(label, np.array2string(values, precision=places,
                                     floatmode="fixed"))

    show("roh      ", raw, 1)
    show("Gewichte ", softmax(raw), 3)
    show("skaliert ", raw / np.sqrt(d), 1)
    show("Gewichte ", softmax(raw / np.sqrt(d)), 3)


if __name__ == "__main__":
    main()
