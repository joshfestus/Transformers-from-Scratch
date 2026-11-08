"""Kapitel 17. Cross-Attention: Queries von hier, Keys von dort.

Self-Attention vergleicht eine Sequenz mit sich selbst. Cross-Attention
vergleicht eine Sequenz mit einer anderen, und so lässt ein
Übersetzungsmodell den Satz, den es schreibt, den Satz befragen, den es
bekommen hat. Die Arithmetik ist identisch. Nur woher die drei
Projektionen kommen, ändert sich.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

rng = np.random.default_rng(0)

SOURCE = ["le", "chat", "a", "bu", "le", "lait"]
TARGET = ["die", "Katze", "trank"]
D_MODEL, D_HEAD = 8, 4


def softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def attend(queries_from, keys_from, w_q, w_k, w_v):
    """Ein Head. Nur die beiden Quellen ändern sich."""
    q = queries_from @ w_q
    k = keys_from @ w_k
    v = keys_from @ w_v
    scores = q @ k.T / np.sqrt(k.shape[-1])
    weights = softmax(scores)
    return weights, weights @ v


def main():
    source = rng.standard_normal((len(SOURCE), D_MODEL))
    target = rng.standard_normal((len(TARGET), D_MODEL))
    w_q = rng.standard_normal((D_MODEL, D_HEAD)) / np.sqrt(D_MODEL)
    w_k = rng.standard_normal((D_MODEL, D_HEAD)) / np.sqrt(D_MODEL)
    w_v = rng.standard_normal((D_MODEL, D_HEAD)) / np.sqrt(D_MODEL)

    print("ZWEI SEQUENZEN VERSCHIEDENER LÄNGE")
    print(f"  Quelle, {len(SOURCE)} Tokens  {SOURCE}")
    print(f"  Ziel,   {len(TARGET)} Tokens  {TARGET}")
    print(f"  Form Quelle {source.shape}   Form Ziel {target.shape}")

    print("\nSELF-ATTENTION AUF DEM ZIEL")
    weights, out = attend(target, target, w_q, w_k, w_v)
    print(f"  Gewichte {weights.shape}   Ausgabe {out.shape}")
    print("  jede Zielposition gegen jede Zielposition")

    print("\nCROSS-ATTENTION, QUERIES DES ZIELS GEGEN KEYS DER QUELLE")
    weights, out = attend(target, source, w_q, w_k, w_v)
    print(f"  Gewichte {weights.shape}   Ausgabe {out.shape}")
    print("  eine Zeile pro Zielposition, eine Spalte pro Quellposition")

    print("\nDIE GEWICHTE, ZEILE FÜR ZEILE")
    header = "        " + "".join(f"{w:>8}" for w in SOURCE)
    print(header)
    for token, row in zip(TARGET, weights):
        print(f"  {token:>6}" + "".join(f"{v:>8.3f}" for v in row))
    print("  Zeilensummen", weights.sum(axis=1))

    print("\nWAS SICH ÄNDERT UND WAS NICHT")
    print("  die Ausgabe hat immer eine Zeile pro Query")
    print("  Cross-Attention liefert also eine Antwort in der Form des")
    print("  Ziels, gebaut ganz aus Values der Quelle")

    print("\nDIE LÄNGE DER AUSGABE FOLGT DEN QUERIES, NICHT DEN KEYS")
    for n_target in [1, 3, 9]:
        t = rng.standard_normal((n_target, D_MODEL))
        _, o = attend(t, source, w_q, w_k, w_v)
        noun = "Query  " if n_target == 1 else "Queries"
        print(f"  {n_target} {noun} gegen {len(SOURCE)} Keys "
              f"ergibt {o.shape}")


if __name__ == "__main__":
    main()
