"""Kapitel 26. Reine Decoder-Modelle: eine Sequenz, viele Lektionen.

Eine kausale Maske heißt, Position i sagt Position i plus eins vorher,
an jeder Position zugleich. Eine Sequenz aus n Tokens liefert also n
Trainingssignale aus einem einzigen Vorwärtsdurchlauf, und diese
Effizienz hat diese Architektur zum Standard gemacht.
"""
import numpy as np

from transformer_numpy import Config, forward, init_params, softmax

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

TOKENS = [3, 14, 7, 1, 22]


def shifted_targets(tokens):
    """Die Antwort jeder Position ist das Token, das ihr folgt."""
    return list(tokens[1:])


def cross_entropy(logits, targets):
    """Mittlere negative Log-Wahrscheinlichkeit des richtigen Tokens."""
    losses = []
    for row, target in zip(logits[:-1], targets):
        probs = softmax(row)
        losses.append(-np.log(probs[target] + 1e-12))
    return float(np.mean(losses)), losses


def main():
    cfg = Config()
    params = init_params(cfg, seed=0)
    logits = forward(TOKENS, params, cfg)

    print("EINE SEQUENZ, VIELE VORHERSAGEN")
    targets = shifted_targets(TOKENS)
    print(f"  Eingabe {TOKENS}")
    print(f"  Ziele   {targets}   eins pro Position außer der letzten")
    print(f"  Logits  {logits.shape}, eine Zeile Scores pro Position")

    print("\nJEDE POSITION SAGT DIE NÄCHSTE VORHER")
    mean, each = cross_entropy(logits, targets)
    for i, (target, loss) in enumerate(zip(targets, each)):
        print(f"  Position {i}, Ziel {target:<4} Loss {loss:.3f}")
    print(f"  mittlerer Loss {mean:.3f}")
    print(f"  Raten ergäbe {np.log(cfg.vocab):.3f} bei einem Vokabular "
          f"von {cfg.vocab}")

    print("\nWARUM DAS EFFIZIENT IST")
    for n in [8, 128, 2048]:
        print(f"  eine Sequenz aus {n:<5} gibt {n - 1:<5} Signale "
              f"aus einem Durchlauf")

    print("\nDER AUSGABE-HEAD IST DIE BREITESTE MATRIX IM MODELL")
    print("  d_model   Vokabular   Ausgabe-Head     ein Block")
    sizes = [(768, 50257, 3072), (4096, 128000, 16384)]
    for d_model, vocab, d_ff in sizes:
        head = d_model * vocab
        block = 4 * d_model ** 2 + 2 * d_model * d_ff
        print(f"  {d_model:<9} {vocab:<10_} {head:>13_} {block:>12_}")
    print("  deshalb wird die Embedding-Tabelle dafür wiederverwendet")

    print("\nWAS DAS MODELL AN DER LETZTEN POSITION LIEFERT")
    probs = softmax(logits[-1])
    order = np.argsort(probs)[::-1][:5]
    for rank, token in enumerate(order, 1):
        share = probs[token]
        print(f"  {rank}. Token {token:<4} Wahrscheinl. {share:.4f}")
    print(f"  die ganze Zeile ergibt {probs.sum():.4f}")


if __name__ == "__main__":
    main()
