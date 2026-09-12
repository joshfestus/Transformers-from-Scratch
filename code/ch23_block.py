"""Chapter 23. One block, assembled, and its parameters counted twice.

Everything in Part IV goes in here: attention from Part III, the
feedforward network, two normalisations and two residual additions. The
formula for the parameter count is derived in the chapter and checked
against the arrays here.
"""
import numpy as np

from transformer_numpy import (Config, attention, feedforward,
                               init_params, layer_norm)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def block_pre_norm(x, p, cfg):
    """Normalise, then the sublayer, then add. The modern arrangement."""
    attended, _ = attention(layer_norm(x, p["ln1_gain"], p["ln1_bias"]),
                            p, cfg)
    x = x + attended
    x = x + feedforward(layer_norm(x, p["ln2_gain"], p["ln2_bias"]), p)
    return x


def block_post_norm(x, p, cfg):
    """The sublayer, then add, then normalise. The 2017 arrangement."""
    attended, _ = attention(x, p, cfg)
    x = layer_norm(x + attended, p["ln1_gain"], p["ln1_bias"])
    x = layer_norm(x + feedforward(x, p),
                   p["ln2_gain"], p["ln2_bias"])
    return x


def block_parameters(d_model, d_ff):
    """Four attention matrices, two feedforward ones, two norms."""
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

    print("ONE BLOCK, TWO ARRANGEMENTS")
    pre = block_pre_norm(x, block, cfg)
    post = block_post_norm(x, block, cfg)
    print(f"  input {x.shape}   pre-norm out {pre.shape}   "
          f"post-norm out {post.shape}")
    print(f"  largest difference between them "
          f"{np.abs(pre - post).max():.3f}")
    print("  same weights, same input, different arrangement,")
    print("  different answer")

    print("\nTHE SHAPE NEVER CHANGES")
    print(f"  in {x.shape} out {pre.shape}, which is what lets blocks")
    print("  be stacked to any depth")

    print("\nPARAMETERS, FROM THE FORMULA AND FROM THE ARRAYS")
    formula = block_parameters(cfg.d_model, cfg.d_ff)
    actual = counted(block)
    print(f"  formula {formula:,}")
    print(f"  counted {actual:,}")
    print(f"  agree: {formula == actual}")

    print("\nTHE SAME FORMULA AT REAL SIZES")
    print("  d_model   d_ff     parameters per block   layers   total")
    for d_model, d_ff, layers in [(768, 3072, 12), (1024, 4096, 24),
                                  (4096, 16384, 32)]:
        each = block_parameters(d_model, d_ff)
        print(f"  {d_model:<9} {d_ff:<8} {each:>20,}   {layers:<8} "
              f"{each * layers / 1e9:.2f}B")

    print("\nWHAT EACH PIECE CONTRIBUTES, THIS BOOK'S MODEL")
    pieces = [("attention", 4 * cfg.d_model ** 2),
              ("feedforward", 2 * cfg.d_model * cfg.d_ff + cfg.d_ff
               + cfg.d_model),
              ("normalisation", 4 * cfg.d_model)]
    for name, count in pieces:
        print(f"  {name:<15} {count:>8,}   {100 * count / formula:.1f}%")


if __name__ == "__main__":
    main()
