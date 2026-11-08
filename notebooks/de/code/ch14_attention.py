"""Kapitel 14. Ein Attention-Head mit Zahlen, die du nachrechnen kannst.

Vier Tokens, eine Modelldimension von vier, ein Head, eine Key-Dimension
von vier. Jedes Gewicht ist hier eine Null oder eine Eins, und jedes
Embedding ist eine kleine ganze Zahl, also lässt sich jedes Produkt mit
dem Bleistift nachrechnen. Die Scores werden durch die Wurzel aus vier
geteilt, also genau durch zwei.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

TOKENS = ["die", "Katze", "trank", "Milch"]

# Eine Zeile pro Token. In einem trainierten Modell werden diese Werte
# gelernt. Hier sind sie so gewählt, dass "Katze" und "Milch" ein Merkmal
# teilen und "trank" nicht.
X = np.array([
    [1.0, 0.0, 1.0, 0.0],   # die
    [0.0, 2.0, 0.0, 1.0],   # Katze
    [1.0, 0.0, 2.0, 0.0],   # trank
    [0.0, 1.0, 0.0, 2.0],   # Milch
])

W_Q = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
])

W_K = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

W_V = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 1.0],
])


def softmax(scores):
    """Softmax je Zeile, ums Zeilenmaximum verschoben: kein Überlauf."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def attention(x, w_q, w_k, w_v):
    """Ein Attention-Head, jeder Zwischenschritt bleibt erhalten."""
    queries = x @ w_q
    keys = x @ w_k
    values = x @ w_v
    d_k = keys.shape[-1]
    raw = queries @ keys.T
    scaled = raw / np.sqrt(d_k)
    weights = softmax(scaled)
    output = weights @ values
    return queries, keys, values, raw, scaled, weights, output


def show(name, matrix, cols):
    """Eine beschriftete Matrix so ausgeben, wie das Buch sie druckt."""
    print(f"\n{name}   Form {matrix.shape}")
    print(" " * 7 + "".join(f"{c:>8}" for c in cols))
    for label, row in zip(TOKENS, matrix):
        print(f"{label:>6} " + "".join(f"{v:>8.3f}" for v in row))


def main():
    q, k, v, raw, scaled, weights, out = attention(X, W_Q, W_K, W_V)
    show("X, die Eingabe-Embeddings", X, ["e0", "e1", "e2", "e3"])
    show("Q = X W_Q", q, ["q0", "q1", "q2", "q3"])
    show("K = X W_K", k, ["k0", "k1", "k2", "k3"])
    show("V = X W_V", v, ["v0", "v1", "v2", "v3"])
    show("Rohe Scores, Q mal K transponiert", raw, TOKENS)
    show("Skalierte Scores, jeder Eintrag halbiert", scaled, TOKENS)
    show("Attention-Gewichte, Softmax je Zeile", weights, TOKENS)
    show("Ausgabe, Gewichte mal V", out, ["o0", "o1", "o2", "o3"])
    print("\nZeilensummen der Gewichte:", weights.sum(axis=1))


if __name__ == "__main__":
    main()
