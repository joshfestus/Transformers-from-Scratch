"""Kapitel 2. Formen, Indizes und die beiden Konventionen.

Die nützlichste Gewohnheit beim Debuggen in diesem Fach ist, eine Form
auszugeben. In dieser Datei geht es darum, eine Form lesen zu lernen.
"""
import numpy as np

np.set_printoptions(precision=2, suppress=True, floatmode="fixed")


def plural(count, one, many):
    return f"{count} {one}" if count == 1 else f"{count} {many}"


def describe(name, array):
    """Die Form eines Arrays ausgeben, und wie viel darin steckt."""
    axes = plural(array.ndim, "Achse", "Achsen")
    print(f"  {name:<12} Form {str(array.shape):<14} "
          f"{axes}, {plural(array.size, 'Zahl', 'Zahlen')}")


def main():
    scalar = np.array(3.0)
    vector = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.array([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]])
    batch = np.zeros((2, 3, 4))

    print("VIER OBJEKTE UND IHRE FORMEN")
    describe("scalar", scalar)
    describe("vector", vector)
    describe("matrix", matrix)
    describe("batch", batch)

    print("\nEINE FORM LESEN, VON LINKS NACH RECHTS")
    print("  (2, 3, 4) heißt 2 Beispiele, jedes mit 3 Positionen,")
    print("  jede Position ein Vektor aus 4 Zahlen")

    print("\nINDIZIEREN IN DIE MATRIX")
    print("  matrix[1]      Zeile 1   ", matrix[1])
    print("  matrix[:, 2]   Spalte 2  ", matrix[:, 2])
    print("  matrix[1, 2]   eine Zahl ", matrix[1, 2])

    print("\nTRANSPONIEREN MACHT AUS ZEILEN SPALTEN")
    describe("matrix", matrix)
    describe("matrix.T", matrix.T)
    print("  matrix.T[2] ist matrix[:, 2] ", matrix.T[2])

    print("\nDIE BEIDEN KONVENTIONEN, DIESELBE ARITHMETIK")
    x_row = np.array([[1.0, 2.0]])          # eine Zeile
    w = np.array([[3.0, 4.0],
                  [5.0, 6.0]])
    print("  Zeile mal Matrix, x W    ", (x_row @ w).ravel())
    print("  Matrix mal Spalte, W x   ", (w.T @ x_row.T).ravel())
    print("  dieselben Zahlen, nur anders angeordnet")


if __name__ == "__main__":
    main()
