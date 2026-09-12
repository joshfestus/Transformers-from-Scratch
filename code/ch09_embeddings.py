"""Chapter 9. Embeddings: the table, and what the numbers in it mean.

Two things are demonstrated here. First that looking a row up in a table
is the same operation as multiplying a one-hot vector by that table,
which is what makes the table trainable. Second that vectors learned
from context end up arranged by context, which is the whole reason the
table is worth learning.
"""
import numpy as np

from ch08_bpe import CORPUS

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def build_vocabulary(text):
    return sorted(set(text.split()))


def one_hot(index, size):
    vector = np.zeros(size)
    vector[index] = 1.0
    return vector


def cooccurrence(text, vocab, window=2):
    """Count how often each pair of words appears near each other."""
    index = {word: i for i, word in enumerate(vocab)}
    counts = np.zeros((len(vocab), len(vocab)))
    words = text.split()
    for position, word in enumerate(words):
        lo = max(0, position - window)
        hi = min(len(words), position + window + 1)
        for other in range(lo, hi):
            if other != position:
                counts[index[word], index[words[other]]] += 1.0
    return counts


def embed_by_svd(counts, dimensions=2):
    """Compress the count table down to a few numbers per word."""
    smoothed = np.log1p(counts)
    centred = smoothed - smoothed.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    return u[:, :dimensions] * s[:dimensions]


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def nearest(vocab, vectors, word, count=3):
    i = vocab.index(word)
    scored = [(cosine(vectors[i], vectors[j]), vocab[j])
              for j in range(len(vocab)) if j != i]
    return sorted(scored, reverse=True)[:count]


def main():
    vocab = build_vocabulary(CORPUS)
    size = len(vocab)
    rng = np.random.default_rng(0)
    table = np.round(rng.standard_normal((size, 4)), 2)

    print("A LOOKUP IS A MATRIX MULTIPLICATION")
    index = vocab.index("cat")
    print(f"  vocabulary of {size} words, embeddings of 4 numbers")
    print(f"  'cat' is index {index}")
    print("  by lookup       ", table[index])
    print("  by one-hot times the table")
    print("                  ", one_hot(index, size) @ table)
    print("  identical:", np.allclose(table[index], one_hot(index, size)
                                      @ table))
    print("  the second form is differentiable, which is why the table")
    print("  can be learned at all")

    print("\nVECTORS LEARNED FROM CONTEXT ALONE")
    counts = cooccurrence(CORPUS, vocab)
    vectors = embed_by_svd(counts, dimensions=2)
    print("  word        two numbers each")
    for word, vector in zip(vocab, vectors):
        print(f"  {word:<11} {vector}")

    print("\nNEAREST NEIGHBOURS BY COSINE SIMILARITY")
    for word in ["cat", "milk", "the"]:
        close = nearest(vocab, vectors, word)
        pairs = ", ".join(f"{w} {s:.2f}" for s, w in close)
        print(f"  {word:<11} {pairs}")

    print("\nCOSINE DOES NOT CARE ABOUT LENGTH")
    a = np.array([1.0, 2.0])
    print(f"  cosine of [1, 2] with [2, 4]       {cosine(a, 2 * a):.3f}")
    product = float(a @ (2 * a))
    print(f"  dot product of the same two pairs  {product:.3f}")
    print("  same direction, different lengths, so the cosine is one")
    print("  and the dot product is not")


if __name__ == "__main__":
    main()
