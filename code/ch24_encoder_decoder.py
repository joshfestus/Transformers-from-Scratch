"""Chapter 24. The 2017 architecture, both stacks and the bridge.

An encoder that reads the whole input at once, a decoder that writes one
token at a time, and a cross-attention layer in every decoder block that
lets the second consult the first.
"""
import numpy as np

from transformer_numpy import (Config, feedforward, layer_norm,
                               positional_encoding, softmax, split_heads,
                               merge_heads, init_params)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def heads_attention(q_from, kv_from, p, cfg, causal=False, prefix=""):
    """One multi-head attention, with the two sources named separately."""
    q = split_heads(q_from @ p[prefix + "w_q"], cfg.n_heads)
    k = split_heads(kv_from @ p[prefix + "w_k"], cfg.n_heads)
    v = split_heads(kv_from @ p[prefix + "w_v"], cfg.n_heads)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(cfg.d_head)
    if causal:
        n = q_from.shape[0]
        allowed = np.tril(np.ones((n, n), dtype=bool))
        scores = np.where(allowed, scores, -np.inf)
    weights = softmax(scores)
    return merge_heads(weights @ v) @ p[prefix + "w_o"], weights


def encoder_block(x, p, cfg):
    """Self-attention with no mask, then a feedforward network."""
    normed = layer_norm(x, p["ln1_gain"], p["ln1_bias"])
    attended, _ = heads_attention(normed, normed, p, cfg)
    x = x + attended
    return x + feedforward(layer_norm(x, p["ln2_gain"], p["ln2_bias"]), p)


def decoder_block(x, memory, p, cross, cfg):
    """Masked self-attention, then cross-attention, then feedforward."""
    normed = layer_norm(x, p["ln1_gain"], p["ln1_bias"])
    attended, _ = heads_attention(normed, normed, p, cfg, causal=True)
    x = x + attended
    normed = layer_norm(x, p["ln2_gain"], p["ln2_bias"])
    crossed, weights = heads_attention(normed, memory, cross, cfg)
    x = x + crossed
    return x + feedforward(layer_norm(x, p["ln2_gain"], p["ln2_bias"]),
                           p), weights


def run(source_tokens, target_tokens, params, cfg):
    """The whole model, encoder then decoder, with shapes recorded."""
    shapes = []
    x = params["embed"][source_tokens]
    x = x + positional_encoding(len(source_tokens), cfg.d_model)
    shapes.append(("source embedded", x.shape))
    for p in params["blocks"]:
        x = encoder_block(x, p, cfg)
    memory = layer_norm(x, params["final_gain"], params["final_bias"])
    shapes.append(("encoder output, the memory", memory.shape))

    y = params["embed"][target_tokens]
    y = y + positional_encoding(len(target_tokens), cfg.d_model)
    shapes.append(("target embedded", y.shape))
    for p, cross in zip(params["blocks"], params["cross"]):
        y, weights = decoder_block(y, memory, p, cross, cfg)
    y = layer_norm(y, params["final_gain"], params["final_bias"])
    logits = y @ params["embed"].T
    shapes.append(("decoder output", y.shape))
    shapes.append(("logits", logits.shape))
    return shapes, weights


def base_parameters(d_model=512, d_ff=2048, layers=6, vocab=37000):
    """The 2017 base model, from the numbers the paper gives."""
    enc = layers * (4 * d_model ** 2 + 2 * d_model * d_ff + d_ff + d_model
                    + 4 * d_model)
    dec = layers * (8 * d_model ** 2 + 2 * d_model * d_ff + d_ff + d_model
                    + 6 * d_model)
    return {"encoder": enc, "decoder": dec, "embedding": vocab * d_model}


def main():
    cfg = Config()
    params = init_params(cfg, seed=0)
    rng = np.random.default_rng(1)
    params["cross"] = [
        {k: rng.standard_normal(v.shape) / np.sqrt(cfg.d_model)
         for k, v in p.items() if k.startswith("w_")}
        for p in params["blocks"]]

    source = [3, 14, 7, 1, 22, 9]
    target = [5, 18, 2]
    shapes, cross_weights = run(source, target, params, cfg)

    print("SHAPES THROUGH THE WHOLE MODEL")
    print(f"  source {len(source)} tokens, target {len(target)} tokens")
    for name, shape in shapes:
        print(f"  {name:<28} {shape}")

    print("\nTHE CROSS-ATTENTION IS THE ONLY RECTANGULAR ONE")
    print(f"  shape {cross_weights.shape}, "
          f"heads by target by source")
    print("  row sums, head 0", cross_weights[0].sum(axis=1))

    print("\nTHE 2017 BASE MODEL, BY COMPONENT")
    counts = base_parameters()
    total = sum(counts.values())
    for name, count in counts.items():
        print(f"  {name:<12} {count:>12,}   {100 * count / total:.0f}%")
    print(f"  {'total':<12} {total:>12,}")
    print("  a decoder block holds eight square matrices, not four,")
    print("  because it has two attention layers")


if __name__ == "__main__":
    main()
