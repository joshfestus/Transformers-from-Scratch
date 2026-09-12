"""Chapter 26. Decoder-only models, and why one sequence is many lessons.

A causal mask means position i predicts position i plus one, at every
position at once. A sequence of n tokens therefore supplies n training
signals from a single forward pass, which is the efficiency that made
this the default architecture.
"""
import numpy as np

from transformer_numpy import Config, forward, init_params, softmax

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

TOKENS = [3, 14, 7, 1, 22]


def shifted_targets(tokens):
    """Every position's answer is the token that follows it."""
    return list(tokens[1:])


def cross_entropy(logits, targets):
    """Average negative log probability of the correct next token."""
    losses = []
    for row, target in zip(logits[:-1], targets):
        probs = softmax(row)
        losses.append(-np.log(probs[target] + 1e-12))
    return float(np.mean(losses)), losses


def main():
    cfg = Config()
    params = init_params(cfg, seed=0)
    logits = forward(TOKENS, params, cfg)

    print("ONE SEQUENCE, MANY PREDICTIONS")
    targets = shifted_targets(TOKENS)
    print(f"  input   {TOKENS}")
    print(f"  targets {targets}   one per position except the last")
    print(f"  logits  {logits.shape}, one row of scores per position")

    print("\nEACH POSITION PREDICTS THE NEXT ONE")
    mean, each = cross_entropy(logits, targets)
    for i, (target, loss) in enumerate(zip(targets, each)):
        print(f"  position {i} predicts {target:<4} loss {loss:.3f}")
    print(f"  mean loss {mean:.3f}")
    print(f"  chance would be {np.log(cfg.vocab):.3f} for a vocabulary "
          f"of {cfg.vocab}")

    print("\nWHY THIS IS EFFICIENT")
    for n in [8, 128, 2048]:
        print(f"  a sequence of {n:<5} gives {n - 1:<5} signals "
              f"from one pass")

    print("\nTHE OUTPUT HEAD IS THE WIDEST MATRIX IN THE MODEL")
    print("  d_model   vocabulary   output head   one block")
    sizes = [(768, 50257, 3072), (4096, 128000, 16384)]
    for d_model, vocab, d_ff in sizes:
        head = d_model * vocab
        block = 4 * d_model ** 2 + 2 * d_model * d_ff
        print(f"  {d_model:<9} {vocab:<10,} {head:>13,} {block:>11,}")
    print("  which is why the embedding table is reused for it")

    print("\nWHAT THE MODEL ACTUALLY PRODUCES AT THE LAST POSITION")
    probs = softmax(logits[-1])
    order = np.argsort(probs)[::-1][:5]
    for rank, token in enumerate(order, 1):
        share = probs[token]
        print(f"  {rank}. token {token:<4} probability {share:.4f}")
    print(f"  the whole row adds to {probs.sum():.4f}")


if __name__ == "__main__":
    main()
