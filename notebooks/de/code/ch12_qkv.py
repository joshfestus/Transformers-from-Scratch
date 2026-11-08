"""Kapitel 12. Query, Key und Value: ein Nachschlagen, das mischen darf.

Ein gewöhnliches Wörterbuch findet einen Key genau oder scheitert. Die
Attention bewertet jeden Key, macht aus den Scores Anteile und gibt eine
Mischung zurück. Diese Datei stellt beides nebeneinander.
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
NAMES = ["Nord", "Ost", "Süd", "West"]


def softmax(scores):
    shifted = scores - scores.max()
    e = np.exp(shifted)
    return e / e.sum()


def exact_lookup(query):
    """Ein Wörterbuch. Ein Key passt genau, oder es kommt nichts."""
    for name, key, value in zip(NAMES, KEYS, VALUES):
        if np.allclose(query, key):
            return name, value
    return None, None


def soft_lookup(query, sharpness=4.0):
    """Jeden Key bewerten, normieren, dann die Values danach mischen."""
    scores = KEYS @ query * sharpness
    weights = softmax(scores)
    return weights, weights @ VALUES


def projections(x, w_q, w_k, w_v):
    """Aus einer Eingabezeile werden drei verschiedene Zeilen."""
    return x @ w_q, x @ w_k, x @ w_v


def main():
    print("EIN GEWÖHNLICHES NACHSCHLAGEN")
    for query in [np.array([1.0, 0.0]), np.array([0.7, 0.7])]:
        name, value = exact_lookup(query)
        found = f"{name}, {value}" if name else "überhaupt nichts"
        print(f"  Query {query}  ->  {found}")

    print("\nDIESELBEN ZWEI QUERIES, BEWERTET UND GEMISCHT")
    for query in [np.array([1.0, 0.0]), np.array([0.7, 0.7])]:
        weights, blended = soft_lookup(query)
        print(f"  Query {query}")
        for name, weight in zip(NAMES, weights):
            print(f"    {name:<7} {weight:.3f}")
        print(f"    Ergebnis {blended}")

    print("\nEINE QUERY, DIE NICHTS TRIFFT, BEKOMMT TROTZDEM ANTWORT")
    weights, blended = soft_lookup(np.array([0.0, 0.0]))
    print("  Query [0. 0.]")
    print("    Gewichte", weights)
    print("    Ergebnis", blended)
    print("  jeder Key bekam null, also mischt die Antwort gleichmäßig")

    print("\nEINE EINGABE, DREI VERSCHIEDENE VEKTOREN")
    x = np.array([1.0, 2.0])
    w_q = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_k = np.array([[0.0, 1.0], [1.0, 0.0]])
    w_v = np.array([[2.0, 0.0], [0.0, 0.5]])
    q, k, v = projections(x, w_q, w_k, w_v)
    print("  Eingabe ", x)
    print("  Query   ", q, " wonach diese Position sucht")
    print("  Key     ", k, " was sie zum Abgleich anbietet")
    print("  Value   ", v, " was sie beiträgt, wenn man auf sie achtet")
    print("  drei Rollen, drei Matrizen, eine Eingabe")


if __name__ == "__main__":
    main()
