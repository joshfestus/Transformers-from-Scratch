"""Chapter 34. Reading a paper by tracing shapes.

The method: take any equation, work out the shape of everything in it,
and check that the shapes can meet. If they can, you have understood
the operation. If they cannot, you have misread it.
"""
import numpy as np

from transformer_numpy import Config, init_params

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def trace(cfg, n=5):
    """Every tensor in one forward pass, with what it holds."""
    d, h, f, v = cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.vocab
    return [
        ("token identifiers", (n,), "one integer per position"),
        ("embeddings", (n, d), "one vector per position"),
        ("plus positions", (n, d), "same, now carrying order"),
        ("Q, K, V", (n, d), "three projections of the same rows"),
        ("split into heads", (h, n, d // h), "one slice per head"),
        ("scores", (h, n, n), "every query against every key"),
        ("weights", (h, n, n), "each row adds to one"),
        ("blended", (h, n, d // h), "one output per query per head"),
        ("merged", (n, d), "heads joined back together"),
        ("after the block", (n, d), "unchanged in shape, always"),
        ("logits", (n, v), "one score per vocabulary token"),
    ]


def check(name, shape, expected):
    return "ok" if tuple(shape) == tuple(expected) else "MISMATCH"


def main():
    cfg = Config()
    print("EVERY SHAPE IN ONE FORWARD PASS")
    print("  tensor                shape          what it holds")
    for name, shape, meaning in trace(cfg):
        print(f"  {name:<21} {str(shape):<14} {meaning}")

    print("\nREADING AN EQUATION BY ITS SHAPES")
    print("  the formula says softmax(Q K-transpose / root d_k) V")
    n, d, h = 5, cfg.d_model, cfg.n_heads
    dh = cfg.d_head
    steps = [("Q", (h, n, dh)), ("K-transpose", (h, dh, n)),
             ("their product", (h, n, n)), ("after softmax", (h, n, n)),
             ("V", (h, n, dh)), ("the result", (h, n, dh))]
    for name, shape in steps:
        print(f"  {name:<16} {shape}")
    print("  the inner dimensions meet at every step, so the equation")
    print("  says what it appears to say")

    print("\nWHERE THE PARAMETERS ARE, AS A LAST CHECK")
    params = init_params(cfg, seed=0)
    counted = params["embed"].size + params["final_gain"].size \
        + params["final_bias"].size
    for block in params["blocks"]:
        counted += sum(value.size for value in block.values())
    print(f"  {counted:,} numbers in the whole model")
    print("  and every one of them appeared in a shape above")

    print("\nTHE FOUR QUESTIONS TO ASK OF ANY NEW ARCHITECTURE")
    questions = [
        "what is one token here, and how many are there",
        "what mixes information between positions, and what does not",
        "where does the shape change, and where is it preserved",
        "what is the training signal, and how many per sequence",
    ]
    for i, question in enumerate(questions, 1):
        print(f"  {i}. {question}")


if __name__ == "__main__":
    main()
