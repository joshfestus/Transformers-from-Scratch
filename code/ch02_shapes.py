"""Chapter 2. Shapes, indexing, and the two conventions.

The single most useful debugging habit in this field is printing a
shape. This file is about learning to read one.
"""
import numpy as np

np.set_printoptions(precision=2, suppress=True, floatmode="fixed")


def plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def describe(name, array):
    """Print an array's shape and how much is in it."""
    axes = plural(array.ndim, "axis").replace("axiss", "axes")
    print(f"  {name:<12} shape {str(array.shape):<14} "
          f"{axes}, {plural(array.size, 'number')}")


def main():
    scalar = np.array(3.0)
    vector = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.array([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]])
    batch = np.zeros((2, 3, 4))

    print("FOUR THINGS AND THEIR SHAPES")
    describe("scalar", scalar)
    describe("vector", vector)
    describe("matrix", matrix)
    describe("batch", batch)

    print("\nREADING A SHAPE, LEFT TO RIGHT")
    print("  (2, 3, 4) means 2 examples, each 3 positions,")
    print("  each position a vector of 4 numbers")

    print("\nINDEXING THE MATRIX")
    print("  matrix[1]      row 1     ", matrix[1])
    print("  matrix[:, 2]   column 2  ", matrix[:, 2])
    print("  matrix[1, 2]   one number", matrix[1, 2])

    print("\nTRANSPOSE TURNS ROWS INTO COLUMNS")
    describe("matrix", matrix)
    describe("matrix.T", matrix.T)
    print("  matrix.T[2] is matrix[:, 2] ", matrix.T[2])

    print("\nTHE TWO CONVENTIONS, SAME ARITHMETIC")
    x_row = np.array([[1.0, 2.0]])          # one row
    w = np.array([[3.0, 4.0],
                  [5.0, 6.0]])
    print("  row times matrix, x W    ", (x_row @ w).ravel())
    print("  matrix times column, W x ", (w.T @ x_row.T).ravel())
    print("  the same numbers, laid out differently")


if __name__ == "__main__":
    main()
