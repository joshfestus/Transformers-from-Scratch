"""Kapitel 3. Matrixmultiplikation, und was sie bedeutet.

Ein Skalarprodukt ist ein Ähnlichkeitswert. Ein Matrixprodukt sind viele
davon auf einmal. Alles, was ein Transformer mit Matrizen tut, ruht auf
diesen beiden Sätzen, und diese Datei misst beide nach.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def dot(a, b):
    """Die Definition, ausgeschrieben. NumPys Version tut dasselbe."""
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def angle_between(a, b):
    """Der Winkel in Grad, aus Skalarprodukt und den beiden Längen."""
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def matmul(a, b):
    """Jede Zeile von a gegen jede Spalte von b, als Schleifen."""
    rows, inner = a.shape
    inner_b, cols = b.shape
    assert inner == inner_b, f"{a.shape} mal {b.shape} geht nicht"
    out = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            out[i, j] = dot(a[i], b[:, j])
    return out


def similarity_demo():
    """Vier Vektoren, und das Skalarprodukt jedes Paares."""
    names = ["Nord", "Nordost", "Ost", "Süd"]
    vectors = np.array([[0.0, 1.0], [0.7071, 0.7071],
                        [1.0, 0.0], [0.0, -1.0]])
    print("  Paar                 Skalar  Winkel")
    reference = vectors[0]
    for name, vector in zip(names, vectors):
        product = np.dot(reference, vector)
        angle = angle_between(reference, vector)
        print(f"  Nord gegen {name:<15} {product:6.3f}"
              f"  {angle:6.1f} Grad")
    return names, vectors


def main():
    print("EIN SKALARPRODUKT MISST ÜBEREINSTIMMUNG")
    similarity_demo()
    print("  ein Skalarprodukt ist am größten, wenn beide gleich zeigen,")
    print("  null im rechten Winkel, negativ in Gegenrichtung")

    print("\nMATRIXMULTIPLIKATION BESTEHT AUS VIELEN SKALARPRODUKTEN")
    a = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    b = np.array([[1.0, 0.0, 2.0],
                  [0.0, 1.0, 3.0]])
    print("  a Form", a.shape, " b Form", b.shape)
    print("  als Schleifen geschrieben")
    print(matmul(a, b))
    print("  der Operator von NumPy")
    print(a @ b)
    print("  identisch:", np.array_equal(matmul(a, b), a @ b))

    print("\nDIE INNEREN DIMENSIONEN MÜSSEN PASSEN")
    try:
        matmul(a, a)
    except AssertionError as problem:
        print("  ", problem)

    print("\nBROADCASTING, DIE DREI FÄLLE, DIE FUNKTIONIEREN")
    rows = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])
    print("  Matrix (2, 3) plus Skalar")
    print(rows + 10.0)
    print("  Matrix (2, 3) plus Zeile (3,), zu jeder Zeile addiert")
    print(rows + np.array([10.0, 20.0, 30.0]))
    print("  Matrix (2, 3) plus Spalte (2, 1), zu jeder Spalte addiert")
    print(rows + np.array([[10.0], [20.0]]))

    print("\nUND DER FALL, DER NICHT FUNKTIONIERT")
    try:
        print(rows + np.array([1.0, 2.0]))
    except ValueError as problem:
        print("  ", problem)


if __name__ == "__main__":
    main()
