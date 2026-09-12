"""Chapter 19. The feedforward network, and where the parameters live.

Two matrices with an activation between them, applied to every position
on its own. It is the plainest component in the block and it holds most
of the model's weights.
"""
import numpy as np

from transformer_numpy import Config, gelu, init_params

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def feedforward(x, p):
    """Widen, bend, narrow. Every position independently."""
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

    print("THE SHAPES, WIDENING AND NARROWING BACK")
    hidden = x @ block["w_in"] + block["b_in"]
    print(f"  input   {x.shape}")
    print(f"  widened {hidden.shape}   times {cfg.d_ff // cfg.d_model}")
    print(f"  output  {feedforward(x, block).shape}")

    print("\nEVERY POSITION IS PROCESSED ON ITS OWN")
    together = feedforward(x, block)
    apart = np.stack([feedforward(row[None, :], block)[0] for row in x])
    print("  all five at once, then one at a time")
    print("  identical:", np.allclose(together, apart))
    shuffled = feedforward(x[::-1], block)[::-1]
    print("  reversed then unreversed, identical:",
          np.allclose(together, shuffled))
    print("  no position can see any other, which is why attention")
    print("  has to exist at all")

    print("\nWHERE THE PARAMETERS ARE, PER BLOCK")
    print("  model                attention   feedforward   share")
    for d_model, d_ff, name in [(24, 96, "this book"),
                                (768, 3072, "small real model"),
                                (4096, 16384, "large one")]:
        att = attention_parameters(d_model)
        ff = feedforward_parameters(d_model, d_ff)
        share = 100 * ff / (att + ff)
        print(f"  {name:<20} {att:>9,}   {ff:>11,}   {share:.0f}%")

    print("\nWHAT THE EXPANSION RATIO COSTS")
    print("  ratio   feedforward parameters   share of the block")
    for ratio in [1, 2, 4, 8]:
        d_model = 768
        ff = feedforward_parameters(d_model, ratio * d_model)
        att = attention_parameters(d_model)
        print(f"  {ratio:<7} {ff:>22,}   {100 * ff / (att + ff):.0f}%")


if __name__ == "__main__":
    main()
