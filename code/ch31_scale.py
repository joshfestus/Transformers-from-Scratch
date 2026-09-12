"""Chapter 31. Counting a model exactly, and what it costs to run.

Every number here comes from a formula applied to published
hyperparameters, not from a table someone else compiled.
"""
# name, layers, width, feedforward width, vocabulary, context, style
# the style says which of the two arrangements this book has covered the
# model uses: "plain" is chapter 23's, "gated" the variant of chapter 19
MODELS = [
    ("small", 12, 768, 3072, 50257, 1024, "plain"),
    ("medium", 24, 1024, 4096, 50257, 1024, "plain"),
    ("large", 36, 1280, 5120, 50257, 1024, "plain"),
    ("seven billion", 32, 4096, 11008, 32000, 4096, "gated"),
]


def block_parameters(d_model, d_ff, style="plain"):
    """Chapter 23's formula, with the gated variant as an option."""
    attention = 4 * d_model * d_model
    if style == "gated":
        feedforward = 3 * d_model * d_ff       # three matrices, no biases
        norms = 2 * d_model                    # RMSNorm keeps a gain only
    else:
        feedforward = 2 * d_model * d_ff + d_ff + d_model
        norms = 4 * d_model                    # two norms, gain and bias
    return attention + feedforward + norms


def total_parameters(layers, d_model, d_ff, vocab, context,
                     style="plain"):
    total = layers * block_parameters(d_model, d_ff, style)
    total += vocab * d_model                   # the embedding table
    if style == "plain":
        total += context * d_model             # learned positions
        total += 2 * d_model                   # final norm, gain and bias
    else:
        total += vocab * d_model               # a separate output head
        total += d_model                       # final RMSNorm
    return total


def memory_for_training(parameters, bytes_per_weight=2):
    """Weights, gradients, and the two averages AdamW keeps."""
    weights = parameters * bytes_per_weight
    gradients = parameters * bytes_per_weight
    adam = parameters * 4 * 2
    master = parameters * 4
    return {"weights": weights, "gradients": gradients,
            "optimizer": adam, "master copy": master}


def flops_per_token(parameters, training=False):
    """Two multiply-adds per parameter forward, three times to train."""
    return 6 * parameters if training else 2 * parameters


def main():
    print("PARAMETERS, FROM THE FORMULA")
    print("  model            layers  width   parameters   published")
    published = ["124M", "355M", "774M", "6.7B"]
    for (name, layers, d, ff, vocab, ctx, style), claim in zip(
            MODELS, published):
        total = total_parameters(layers, d, ff, vocab, ctx, style)
        print(f"  {name:<16} {layers:<7} {d:<7} {total / 1e6:>9,.1f}M "
              f"{claim:>11}")

    print("\nWHERE THEY ARE, AS THE MODEL GROWS")
    print("  model            blocks   embedding   blocks' share")
    for name, layers, d, ff, vocab, ctx, style in MODELS:
        blocks = layers * block_parameters(d, ff, style)
        embed = vocab * d
        share = 100 * blocks / (blocks + embed)
        print(f"  {name:<16} {blocks / 1e6:>7,.0f}M "
              f"{embed / 1e6:>9,.0f}M {share:>13.0f}%")
    print("  the embedding is most of a small model and almost none")
    print("  of a large one, because it grows with width and the blocks")
    print("  grow with width squared")

    print("\nWHAT IT TAKES TO TRAIN, PER PARAMETER SET")
    name, layers, d, ff, vocab, ctx, style = MODELS[-1]
    total = total_parameters(layers, d, ff, vocab, ctx, style)
    print(f"  the {name} model, {total / 1e9:.2f} billion parameters")
    parts = memory_for_training(total)
    for label, size in parts.items():
        print(f"  {label:<14} {size / 1e9:>6.1f} GB")
    print(f"  {'total':<14} {sum(parts.values()) / 1e9:>6.1f} GB, before "
          f"a single activation")

    print("\nARITHMETIC PER TOKEN")
    print("  model            inference        training")
    for name, layers, d, ff, vocab, ctx, style in MODELS:
        total = total_parameters(layers, d, ff, vocab, ctx, style)
        print(f"  {name:<16} {flops_per_token(total) / 1e9:>7.2f} GFLOP "
              f"{flops_per_token(total, True) / 1e9:>9.2f} GFLOP")

    print("\nWHAT A TRAINING RUN COSTS IN ARITHMETIC")
    total = total_parameters(32, 4096, 11008, 32000, 4096, "gated")
    for tokens in [1e9, 1e11, 1e12]:
        work = flops_per_token(total, True) * tokens
        print(f"  {tokens:>7.0e} tokens   {work:.2e} operations")


if __name__ == "__main__":
    main()
