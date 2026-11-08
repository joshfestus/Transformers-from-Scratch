"""Kapitel 23. Ein Block, zusammengebaut, Parameter doppelt gezählt.

Alles aus Teil IV kommt hier hinein: die Attention aus Teil III, das
Feedforward-Netz, zwei Normierungen und zwei Residual-Additionen. Die
Formel für die Parameterzahl wird im Kapitel hergeleitet und hier an den
Arrays geprüft.
"""
import numpy as np

from transformer_numpy import (Config, attention, feedforward,
                               init_params, layer_norm)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def block_pre_norm(x, p, cfg):
    """Normieren, Teilschicht, addieren. Die moderne Anordnung."""
    attended, _ = attention(layer_norm(x, p["ln1_gain"], p["ln1_bias"]),
                            p, cfg)
    x = x + attended
    x = x + feedforward(layer_norm(x, p["ln2_gain"], p["ln2_bias"]), p)
    return x


def block_post_norm(x, p, cfg):
    """Teilschicht, addieren, normieren. Die Anordnung von 2017."""
    attended, _ = attention(x, p, cfg)
    x = layer_norm(x + attended, p["ln1_gain"], p["ln1_bias"])
    x = layer_norm(x + feedforward(x, p),
                   p["ln2_gain"], p["ln2_bias"])
    return x


def block_parameters(d_model, d_ff):
    """Vier Attention-Matrizen, zwei für Feedforward, zwei Normen."""
    attention_part = 4 * d_model * d_model
    feedforward_part = 2 * d_model * d_ff + d_ff + d_model
    norm_part = 4 * d_model
    return attention_part + feedforward_part + norm_part


def counted(params):
    return sum(v.size for v in params.values())


def main():
    cfg = Config()
    params = init_params(cfg, seed=0)
    block = params["blocks"][0]
    x = params["embed"][[3, 14, 7, 1, 22]]

    print("EIN BLOCK, ZWEI ANORDNUNGEN")
    pre = block_pre_norm(x, block, cfg)
    post = block_post_norm(x, block, cfg)
    print(f"  Eingabe {x.shape}   Pre-Norm aus {pre.shape}   "
          f"Post-Norm aus {post.shape}")
    print(f"  größter Unterschied zwischen beiden "
          f"{np.abs(pre - post).max():.3f}")
    print("  gleiche Gewichte, gleiche Eingabe, andere Anordnung,")
    print("  andere Antwort")

    print("\nDIE FORM ÄNDERT SICH NIE")
    print(f"  ein {x.shape} aus {pre.shape}, und deshalb lassen sich")
    print("  Blöcke beliebig tief stapeln")

    print("\nPARAMETER, AUS DER FORMEL UND AUS DEN ARRAYS")
    formula = block_parameters(cfg.d_model, cfg.d_ff)
    actual = counted(block)
    print(f"  Formel  {formula:_}")
    print(f"  gezählt {actual:_}")
    print(f"  gleich: {formula == actual}")

    print("\nDIESELBE FORMEL BEI ECHTEN GRÖSSEN")
    print("  d_model   d_ff     Parameter pro Block    Schichten Summe")
    for d_model, d_ff, layers in [(768, 3072, 12), (1024, 4096, 24),
                                  (4096, 16384, 32)]:
        each = block_parameters(d_model, d_ff)
        print(f"  {d_model:<9} {d_ff:<8} {each:>20_}   {layers:<9} "
              f"{each * layers / 1e9:.2f} Mrd.")

    print("\nWAS JEDES TEIL BEITRÄGT, DAS MODELL DIESES BUCHES")
    pieces = [("Attention", 4 * cfg.d_model ** 2),
              ("Feedforward", 2 * cfg.d_model * cfg.d_ff + cfg.d_ff
               + cfg.d_model),
              ("Normierung", 4 * cfg.d_model)]
    for name, count in pieces:
        print(f"  {name:<15} {count:>8_}   {100 * count / formula:.1f} %")


if __name__ == "__main__":
    main()
