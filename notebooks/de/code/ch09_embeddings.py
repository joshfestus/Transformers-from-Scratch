"""Kapitel 9. Embeddings: die Tabelle, und was ihre Zahlen bedeuten.

Hier wird zweierlei gezeigt. Erstens, dass eine Zeile in einer Tabelle
nachzuschlagen dieselbe Operation ist, wie einen One-Hot-Vektor mit der
Tabelle zu multiplizieren, und das macht die Tabelle trainierbar.
Zweitens, dass Vektoren, die aus dem Kontext gelernt werden, nach dem
Kontext geordnet sind, und das ist der ganze Grund, sie zu lernen.
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
    """Zählen, wie oft jedes Wortpaar nah beieinander vorkommt."""
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
    """Die Zähltabelle auf wenige Zahlen pro Wort verdichten."""
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

    print("NACHSCHLAGEN IST EINE MATRIXMULTIPLIKATION")
    index = vocab.index("katze")
    print(f"  Vokabular aus {size} Wörtern, Embeddings aus 4 Zahlen")
    print(f"  'katze' hat den Index {index}")
    print("  nachgeschlagen  ", table[index])
    print("  One-Hot mal Tabelle")
    print("                  ", one_hot(index, size) @ table)
    print("  identisch:", np.allclose(table[index], one_hot(index, size)
                                      @ table))
    print("  die zweite Form ist differenzierbar, und deshalb lässt sich")
    print("  die Tabelle überhaupt lernen")

    print("\nVEKTOREN, NUR AUS DEM KONTEXT GELERNT")
    counts = cooccurrence(CORPUS, vocab)
    vectors = embed_by_svd(counts, dimensions=2)
    print("  Wort        je zwei Zahlen")
    for word, vector in zip(vocab, vectors):
        print(f"  {word:<11} {vector}")

    print("\nNÄCHSTE NACHBARN NACH KOSINUS-ÄHNLICHKEIT")
    for word in ["katze", "milch", "die"]:
        close = nearest(vocab, vectors, word)
        pairs = ", ".join(f"{w} {s:.2f}" for s, w in close)
        print(f"  {word:<11} {pairs}")

    print("\nDEM KOSINUS IST DIE LÄNGE EGAL")
    a = np.array([1.0, 2.0])
    print(f"  Kosinus von [1, 2] mit [2, 4]       {cosine(a, 2 * a):.3f}")
    product = float(a @ (2 * a))
    print(f"  Skalarprodukt derselben beiden      {product:.3f}")
    print("  gleiche Richtung, verschiedene Längen, also ist der Kosinus")
    print("  eins und das Skalarprodukt nicht")


if __name__ == "__main__":
    main()
