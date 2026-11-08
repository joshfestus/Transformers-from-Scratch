"""Kapitel 19. Das Feedforward-Netz, und wo die Parameter wohnen.

Zwei Matrizen mit einer Aktivierung dazwischen, auf jede Position für
sich angewendet. Es ist das schlichteste Bauteil im Block, und es
enthält die meisten Gewichte des Modells.
"""
import numpy as np

from transformer_numpy import Config, gelu, init_params

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def feedforward(x, p):
    """Verbreitern, knicken, verengen. Jede Position für sich."""
    return gelu(x @ p["w_in"] + p["b_in"]) @ p["w_out"] + p["b_out"]


def attention_parameters(d_model):
    return 4 * d_model * d_model


def feedforward_parameters(d_model, d_ff):
    return 2 * d_model * d_ff + d_ff + d_model


def main():
    cfg = Config()
    params = init_params(cfg, seed=0)
    block = params["blocks"][0]
    x = params["embed"][[3, 14, 7, 1, 22]]

    print("DIE FORMEN, VERBREITERN UND WIEDER VERENGEN")
    hidden = x @ block["w_in"] + block["b_in"]
    print(f"  Eingabe      {x.shape}")
    ratio = cfg.d_ff // cfg.d_model
    print(f"  verbreitert  {hidden.shape}   mal {ratio}")
    print(f"  Ausgabe      {feedforward(x, block).shape}")

    print("\nJEDE POSITION WIRD FÜR SICH VERARBEITET")
    together = feedforward(x, block)
    apart = np.stack([feedforward(row[None, :], block)[0] for row in x])
    print("  alle fünf auf einmal, dann eine nach der anderen")
    print("  identisch:", np.allclose(together, apart))
    shuffled = feedforward(x[::-1], block)[::-1]
    print("  umgedreht und zurückgedreht, identisch:",
          np.allclose(together, shuffled))
    print("  keine Position sieht eine andere, und deshalb muss es")
    print("  die Attention überhaupt geben")

    print("\nWO DIE PARAMETER SIND, PRO BLOCK")
    print("  Modell                 Attention   Feedforward   Anteil")
    for d_model, d_ff, name in [(24, 96, "dieses Buch"),
                                (768, 3072, "kleines echtes Modell"),
                                (4096, 16384, "großes")]:
        att = attention_parameters(d_model)
        ff = feedforward_parameters(d_model, d_ff)
        share = 100 * ff / (att + ff)
        print(f"  {name:<22}{att:>10_}   {ff:>11_}   {share:.0f} %")

    print("\nWAS DER VERBREITERUNGSFAKTOR KOSTET")
    print("  Faktor  Feedforward-Parameter    Anteil am Block")
    for ratio in [1, 2, 4, 8]:
        d_model = 768
        ff = feedforward_parameters(d_model, ratio * d_model)
        att = attention_parameters(d_model)
        print(f"  {ratio:<7} {ff:>22_}   {100 * ff / (att + ff):.0f} %")


if __name__ == "__main__":
    main()
