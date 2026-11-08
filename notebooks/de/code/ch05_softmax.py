"""Kapitel 5. Aus Scores werden Wahrscheinlichkeiten, ohne Überlauf.

Die Softmax kommt in einem Transformer dreimal vor: in jedem Attention-
Head, im Ausgabe-Head und im Loss. In dieser Datei geht es um die eine
Funktion und den einen Trick, der sie vor dem Zusammenbruch bewahrt.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


def naive_softmax(scores):
    """Die Definition, genau wie geschrieben. Sie läuft über."""
    e = np.exp(scores)
    return e / e.sum()


def softmax(scores, temperature=1.0):
    """Dieselbe Funktion, vorher um das Maximum verschoben."""
    scaled = np.asarray(scores, dtype=float) / temperature
    shifted = scaled - scaled.max()
    e = np.exp(shifted)
    return e / e.sum()


def temperature_table(scores, temperatures):
    return [(t, softmax(scores, t)) for t in temperatures]


def main():
    scores = np.array([2.0, 1.0, 0.1, -1.0])
    print("AUS VIER SCORES WERDEN VIER WAHRSCHEINLICHKEITEN")
    print("  Scores              ", scores)
    print("  Wahrscheinlichkeiten", softmax(scores))
    print("  Summe               ", softmax(scores).sum())

    print("\nDIE REIHENFOLGE BLEIBT, NUR DIE ABSTÄNDE ÄNDERN SICH")
    for t, p in temperature_table(scores, [0.25, 0.5, 1.0, 2.0, 8.0]):
        print(f"  Temperatur {t:<5} {p}  oben {p.max():.3f}")

    print("\nWARUM DIE VERSCHIEBUNG NÖTIG IST")
    big = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = naive_softmax(big)
    print("  Scores        ", big)
    print("  ohne Versch.  ", naive)
    print("  mit Versch.   ", softmax(big))
    print("  mathematisch ändert die Verschiebung nichts, denn")
    print("  jeder Exponentialwert wird durch dieselbe Konstante geteilt")

    print("\nUND DAS ANDERE ENDE DES BEREICHS")
    small = np.array([-1000.0, -1001.0, -1002.0])
    with np.errstate(under="ignore", invalid="ignore"):
        naive = naive_softmax(small)
    print("  Scores        ", small)
    print("  ohne Versch.  ", naive)
    print("  mit Versch.   ", softmax(small))


if __name__ == "__main__":
    main()
