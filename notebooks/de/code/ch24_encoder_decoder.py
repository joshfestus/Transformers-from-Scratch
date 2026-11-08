"""Kapitel 24. Die Architektur von 2017, beide Stapel und die Brücke.

Ein Encoder, der die ganze Eingabe auf einmal liest, ein Decoder, der ein
Token nach dem anderen schreibt, und in jedem Decoder-Block eine
Cross-Attention-Schicht, mit der der zweite den ersten befragt.
"""
import numpy as np

from transformer_numpy import (Config, feedforward, layer_norm,
                               positional_encoding, softmax, split_heads,
                               merge_heads, init_params)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def heads_attention(q_from, kv_from, p, cfg, causal=False, prefix=""):
    """Eine Multi-Head Attention, mit getrennt benannten Quellen."""
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
    """Self-Attention ohne Maske, dann ein Feedforward-Netz."""
    normed = layer_norm(x, p["ln1_gain"], p["ln1_bias"])
    attended, _ = heads_attention(normed, normed, p, cfg)
    x = x + attended
    return x + feedforward(layer_norm(x, p["ln2_gain"], p["ln2_bias"]), p)


def decoder_block(x, memory, p, cross, cfg):
    """Maskierte Self-Attention, Cross-Attention, dann Feedforward."""
    normed = layer_norm(x, p["ln1_gain"], p["ln1_bias"])
    attended, _ = heads_attention(normed, normed, p, cfg, causal=True)
    x = x + attended
    normed = layer_norm(x, p["ln2_gain"], p["ln2_bias"])
    crossed, weights = heads_attention(normed, memory, cross, cfg)
    x = x + crossed
    return x + feedforward(layer_norm(x, p["ln2_gain"], p["ln2_bias"]),
                           p), weights


def run(source_tokens, target_tokens, params, cfg):
    """Das ganze Modell, Encoder, dann Decoder, mit allen Formen."""
    shapes = []
    x = params["embed"][source_tokens]
    x = x + positional_encoding(len(source_tokens), cfg.d_model)
    shapes.append(("Quelle eingebettet", x.shape))
    for p in params["blocks"]:
        x = encoder_block(x, p, cfg)
    memory = layer_norm(x, params["final_gain"], params["final_bias"])
    shapes.append(("Encoder-Ausgabe, das Gedächtnis", memory.shape))

    y = params["embed"][target_tokens]
    y = y + positional_encoding(len(target_tokens), cfg.d_model)
    shapes.append(("Ziel eingebettet", y.shape))
    for p, cross in zip(params["blocks"], params["cross"]):
        y, weights = decoder_block(y, memory, p, cross, cfg)
    y = layer_norm(y, params["final_gain"], params["final_bias"])
    logits = y @ params["embed"].T
    shapes.append(("Decoder-Ausgabe", y.shape))
    shapes.append(("Logits", logits.shape))
    return shapes, weights


def base_parameters(d_model=512, d_ff=2048, layers=6, vocab=37000):
    """Das Basismodell von 2017, aus den Zahlen des Papers."""
    enc = layers * (4 * d_model ** 2 + 2 * d_model * d_ff + d_ff + d_model
                    + 4 * d_model)
    dec = layers * (8 * d_model ** 2 + 2 * d_model * d_ff + d_ff + d_model
                    + 6 * d_model)
    return {"Encoder": enc, "Decoder": dec, "Embedding": vocab * d_model}


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

    print("FORMEN DURCH DAS GANZE MODELL")
    print(f"  Quelle {len(source)} Tokens, Ziel {len(target)} Tokens")
    for name, shape in shapes:
        print(f"  {name:<32} {shape}")

    print("\nDIE CROSS-ATTENTION IST DIE EINZIGE RECHTECKIGE")
    print(f"  Form {cross_weights.shape}, "
          f"Heads mal Ziel mal Quelle")
    print("  Zeilensummen, Head 0", cross_weights[0].sum(axis=1))

    print("\nDAS BASISMODELL VON 2017, NACH BAUTEILEN")
    counts = base_parameters()
    total = sum(counts.values())
    for name, count in counts.items():
        print(f"  {name:<12} {count:>12_}   {100 * count / total:.0f} %")
    print(f"  {'Summe':<12} {total:>12_}")
    print("  ein Decoder-Block enthält acht quadratische Matrizen, nicht")
    print("  vier, weil er zwei Attention-Schichten hat")


if __name__ == "__main__":
    main()
