"""Chapter 3. Matrix multiplication, and what it means.

A dot product is a similarity score. A matrix product is many of them
at once. Everything a transformer does with matrices rests on those two
sentences, and this file measures both.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def dot(a, b):
    """The definition, written out. NumPy's version does the same."""
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def angle_between(a, b):
    """The angle in degrees, from the dot product and the two lengths."""
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def matmul(a, b):
    """Every row of a against every column of b, written as loops."""
    rows, inner = a.shape
    inner_b, cols = b.shape
    assert inner == inner_b, f"cannot multiply {a.shape} by {b.shape}"
    out = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            out[i, j] = dot(a[i], b[:, j])
    return out


def similarity_demo():
    """Four vectors, and the dot product of each pair."""
    names = ["north", "north-east", "east", "south"]
    vectors = np.array([[0.0, 1.0], [0.7071, 0.7071],
                        [1.0, 0.0], [0.0, -1.0]])
    print("  pair                 dot     angle")
    reference = vectors[0]
    for name, vector in zip(names, vectors):
        product = np.dot(reference, vector)
        angle = angle_between(reference, vector)
        print(f"  north against {name:<12} {product:6.3f}"
              f"  {angle:6.1f} degrees")
    return names, vectors


def main():
    print("A DOT PRODUCT MEASURES AGREEMENT")
    similarity_demo()
    print("  a dot product is largest when both point the same way,")
    print("  nought when they are at right angles, negative when opposed")

    print("\nMATRIX MULTIPLICATION IS MANY DOT PRODUCTS")
    a = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    b = np.array([[1.0, 0.0, 2.0],
                  [0.0, 1.0, 3.0]])
    print("  a shape", a.shape, " b shape", b.shape)
    print("  written as loops")
    print(matmul(a, b))
    print("  numpy's operator")
    print(a @ b)
    print("  identical:", np.array_equal(matmul(a, b), a @ b))

    print("\nTHE INNER DIMENSIONS MUST MATCH")
    try:
        matmul(a, a)
    except AssertionError as problem:
        print("  ", problem)

    print("\nBROADCASTING, THE THREE CASES THAT WORK")
    rows = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]])
    print("  matrix (2, 3) plus scalar")
    print(rows + 10.0)
    print("  matrix (2, 3) plus row (3,), added to every row")
    print(rows + np.array([10.0, 20.0, 30.0]))
    print("  matrix (2, 3) plus column (2, 1), added to every column")
    print(rows + np.array([[10.0], [20.0]]))

    print("\nAND THE CASE THAT DOES NOT")
    try:
        print(rows + np.array([1.0, 2.0]))
    except ValueError as problem:
        print("  ", problem)


if __name__ == "__main__":
    main()
