"""Kapitel 15. Mehrere Heads, und das Umformen, das sie billig macht.

Multi-Head Attention sind nicht mehrere Attention-Schichten. Es ist ein
Satz Projektionen, dessen Ausgabe in Gruppen geschnitten wird. Jede
Gruppe macht ihre eigene Attention, und die Ergebnisse werden wieder
zusammengefügt. Die Zahl der Parameter ist bei einem Head dieselbe wie
bei sechzehn.
"""
import numpy as np

from transformer_numpy import (Config, attention, init_params,
                               merge_heads, split_heads)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def shapes_through(x, cfg):
    """Jede Form auf dem Weg durch eine Multi-Head Attention."""
    steps = [("Eingabe", x.shape)]
    q = x @ np.zeros((cfg.d_model, cfg.d_model))
    steps.append(("nach der Query-Projektion", q.shape))
    heads = split_heads(q, cfg.n_heads)
    steps.append(("in Heads aufgeteilt", heads.shape))
    scores = heads @ heads.transpose(0, 2, 1)
    steps.append(("Scores, ein Quadrat pro Head", scores.shape))
    blended = scores @ heads
    steps.append(("gemischt, noch pro Head", blended.shape))
    steps.append(("wieder zusammengefügt", merge_heads(blended).shape))
    return steps


def head_parameters(d_model, n_heads):
    """Vier quadratische Matrizen, egal wie viele Heads."""
    return 4 * d_model * d_model


def main():
    cfg = Config()
    params = init_params(cfg, seed=0)
    x = params["embed"][[3, 14, 7, 1, 22]]

    print(f"EINE ATTENTION-SCHICHT, {cfg.n_heads} HEADS ZU {cfg.d_head}")
    for name, shape in shapes_through(x, cfg):
        print(f"  {name:<30} {shape}")

    print("\nDIE ZAHL DER HEADS ÄNDERT DIE PARAMETERZAHL NICHT")
    print("  Heads   Head-Breite  Parameter")
    for heads in [1, 2, 4, 8, 24]:
        if cfg.d_model % heads:
            continue
        print(f"  {heads:<7} {cfg.d_model // heads:<12} "
              f"{head_parameters(cfg.d_model, heads):_}")
    print("  die Projektionen sind gleich groß, egal wie geschnitten")

    print("\nVOR DEM TRAINING SIEHT JEDER HEAD GLEICH AUS")
    block = params["blocks"][0]
    _, weights = attention(x, block, cfg)
    for head in range(cfg.n_heads):
        row = weights[head][-1]
        print(f"  Head {head}  {row}")
    flat = weights[:, -1, :]
    gaps = [float(np.abs(flat[a] - flat[b]).max())
            for a in range(cfg.n_heads)
            for b in range(a + 1, cfg.n_heads)]
    spread = max(gaps)
    print(f"  größter Abstand zwischen zwei Heads: {spread:.3f}")
    print("  die Gewichte sind klein und zufällig, also ist jeder Score")
    print("  nahe null, und jede Softmax kommt flach heraus")

    print("\nHEADS UNTERSCHEIDEN SICH NUR DURCH DAS TRAINING")
    built = {k: v.copy() for k, v in block.items()}
    built["w_q"] = np.zeros((cfg.d_model, cfg.d_model))
    built["w_k"] = np.zeros((cfg.d_model, cfg.d_model))
    # Head 0 liest Merkmal 2, Head 1 Merkmal 5, beide verstärkt
    built["w_q"][2, 0] = built["w_k"][2, 0] = 16.0
    built["w_q"][5, cfg.d_head] = built["w_k"][5, cfg.d_head] = 16.0
    _, made = attention(x, built, cfg)
    for head in range(2):
        row = made[head][-1]
        print(f"  Head {head}  {row}  Spitze bei Position {row.argmax()}")
    print("  diese beiden Matrizen sind von Hand gesetzt, nicht gelernt,")
    print("  um zu zeigen, dass Heads überhaupt verschieden sein können")


if __name__ == "__main__":
    main()
