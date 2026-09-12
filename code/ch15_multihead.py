"""Chapter 15. Several heads, and the reshaping that makes them cheap.

Multi-head attention is not several attention layers. It is one set of
projections whose output is sliced into groups, each group attending on
its own, and the results joined back together. The parameter count is
the same whether there is one head or sixteen.
"""
import numpy as np

from transformer_numpy import (Config, attention, init_params,
                               merge_heads, split_heads)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def shapes_through(x, cfg):
    """Every shape on the way through one multi-head attention."""
    steps = [("input", x.shape)]
    q = x @ np.zeros((cfg.d_model, cfg.d_model))
    steps.append(("after the query projection", q.shape))
    heads = split_heads(q, cfg.n_heads)
    steps.append(("split into heads", heads.shape))
    scores = heads @ heads.transpose(0, 2, 1)
    steps.append(("scores, one square per head", scores.shape))
    blended = scores @ heads
    steps.append(("blended, still per head", blended.shape))
    steps.append(("merged back together", merge_heads(blended).shape))
    return steps


def head_parameters(d_model, n_heads):
    """Four square matrices, whatever the head count."""
    return 4 * d_model * d_model


def main():
    cfg = Config()
    params = init_params(cfg, seed=0)
    x = params["embed"][[3, 14, 7, 1, 22]]

    print(f"ONE ATTENTION LAYER, {cfg.n_heads} HEADS OF {cfg.d_head}")
    for name, shape in shapes_through(x, cfg):
        print(f"  {name:<28} {shape}")

    print("\nTHE HEAD COUNT DOES NOT CHANGE THE PARAMETER COUNT")
    print("  heads   head width   parameters")
    for heads in [1, 2, 4, 8, 24]:
        if cfg.d_model % heads:
            continue
        print(f"  {heads:<7} {cfg.d_model // heads:<12} "
              f"{head_parameters(cfg.d_model, heads):,}")
    print("  the projections are the same size however they are sliced")

    print("\nBEFORE TRAINING, EVERY HEAD LOOKS THE SAME")
    block = params["blocks"][0]
    _, weights = attention(x, block, cfg)
    for head in range(cfg.n_heads):
        row = weights[head][-1]
        print(f"  head {head}  {row}")
    flat = weights[:, -1, :]
    gaps = [float(np.abs(flat[a] - flat[b]).max())
            for a in range(cfg.n_heads)
            for b in range(a + 1, cfg.n_heads)]
    spread = max(gaps)
    print(f"  the largest gap between any two heads is {spread:.3f}")
    print("  the weights are small and random, so every score is near")
    print("  nought and every softmax comes out flat")

    print("\nHEADS DIFFER ONLY BECAUSE TRAINING MAKES THEM DIFFER")
    built = {k: v.copy() for k, v in block.items()}
    built["w_q"] = np.zeros((cfg.d_model, cfg.d_model))
    built["w_k"] = np.zeros((cfg.d_model, cfg.d_model))
    # head 0 is made to read feature 2, head 1 feature 5, both amplified
    built["w_q"][2, 0] = built["w_k"][2, 0] = 16.0
    built["w_q"][5, cfg.d_head] = built["w_k"][5, cfg.d_head] = 16.0
    _, made = attention(x, built, cfg)
    for head in range(2):
        row = made[head][-1]
        print(f"  head {head}  {row}  peak at position {row.argmax()}")
    print("  these two matrices were written by hand, not learned,")
    print("  to show that the mechanism allows heads to disagree at all")


if __name__ == "__main__":
    main()
