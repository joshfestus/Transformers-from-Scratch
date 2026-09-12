"""Chapter 28. A complete transformer, built from the chapters behind it.

A decoder-only model: token embeddings, sinusoidal positions, a stack of
pre-norm blocks each holding causal multi-head attention and a
feedforward network, a final normalisation, and an output head back to
the vocabulary. Every function here appeared in an earlier chapter, and
nothing in it imports a framework.

The whole thing is one forward pass and about a hundred lines.
"""
import numpy as np


class Config:
    """The shape of the model, in one object."""

    def __init__(self, vocab=32, d_model=24, n_heads=4, n_layers=2,
                 d_ff=96, context=16):
        self.vocab = vocab
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.context = context
        assert d_model % n_heads == 0, "heads must divide the model width"
        self.d_head = d_model // n_heads

    def __repr__(self):
        return (f"Config(vocab={self.vocab}, d_model={self.d_model}, "
                f"n_heads={self.n_heads}, n_layers={self.n_layers}, "
                f"d_ff={self.d_ff})")


def init_params(cfg, seed=0):
    """Small random weights, one dictionary per block. Chapter 7 explains
    why the scale of the initial values matters."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(cfg.d_model)

    def normal(*shape):
        return rng.standard_normal(shape) * scale

    params = {"embed": normal(cfg.vocab, cfg.d_model),
              "final_gain": np.ones(cfg.d_model),
              "final_bias": np.zeros(cfg.d_model),
              "blocks": []}
    for _ in range(cfg.n_layers):
        params["blocks"].append({
            "ln1_gain": np.ones(cfg.d_model),
            "ln1_bias": np.zeros(cfg.d_model),
            "w_q": normal(cfg.d_model, cfg.d_model),
            "w_k": normal(cfg.d_model, cfg.d_model),
            "w_v": normal(cfg.d_model, cfg.d_model),
            "w_o": normal(cfg.d_model, cfg.d_model),
            "ln2_gain": np.ones(cfg.d_model),
            "ln2_bias": np.zeros(cfg.d_model),
            "w_in": normal(cfg.d_model, cfg.d_ff),
            "b_in": np.zeros(cfg.d_ff),
            "w_out": normal(cfg.d_ff, cfg.d_model),
            "b_out": np.zeros(cfg.d_model),
        })
    return params


def positional_encoding(n, d_model):
    """The sinusoidal ladder from chapter 10. No parameters to learn."""
    position = np.arange(n)[:, None]
    index = np.arange(0, d_model, 2)[None, :]
    rate = np.exp(-np.log(10000.0) * index / d_model)
    encoding = np.zeros((n, d_model))
    encoding[:, 0::2] = np.sin(position * rate)
    encoding[:, 1::2] = np.cos(position * rate)
    return encoding


def layer_norm(x, gain, bias, eps=1e-5):
    """Chapter 21. Normalise across the model dimension, then rescale."""
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * gain + bias


def gelu(x):
    """Chapter 4, in the tanh form every implementation actually uses."""
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def softmax(x):
    """Chapter 5. Shift by the row max so it cannot overflow."""
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(n):
    """Chapter 16. True where a query is allowed to look."""
    return np.tril(np.ones((n, n), dtype=bool))


def split_heads(x, n_heads):
    """(n, d_model) becomes (n_heads, n, d_head). Chapter 15."""
    n, d_model = x.shape
    return x.reshape(n, n_heads, d_model // n_heads).transpose(1, 0, 2)


def merge_heads(x):
    """(n_heads, n, d_head) becomes (n, d_model). Chapter 15."""
    n_heads, n, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(n, n_heads * d_head)


def attention(x, p, cfg):
    """Causal multi-head attention. Chapters 13 to 16."""
    q = split_heads(x @ p["w_q"], cfg.n_heads)
    k = split_heads(x @ p["w_k"], cfg.n_heads)
    v = split_heads(x @ p["w_v"], cfg.n_heads)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(cfg.d_head)
    scores = np.where(causal_mask(x.shape[0]), scores, -np.inf)
    weights = softmax(scores)
    return merge_heads(weights @ v) @ p["w_o"], weights


def feedforward(x, p):
    """Chapter 19. Widen, bend, narrow."""
    return gelu(x @ p["w_in"] + p["b_in"]) @ p["w_out"] + p["b_out"]


def block(x, p, cfg):
    """One pre-norm block. Chapters 20, 21 and 23."""
    attended, weights = attention(layer_norm(x, p["ln1_gain"],
                                             p["ln1_bias"]), p, cfg)
    x = x + attended
    x = x + feedforward(layer_norm(x, p["ln2_gain"], p["ln2_bias"]), p)
    return x, weights


def forward(tokens, params, cfg, keep_weights=False):
    """Token identifiers in, one row of logits per position out."""
    assert len(tokens) <= cfg.context, "sequence longer than the context"
    x = params["embed"][tokens]
    x = x + positional_encoding(len(tokens), cfg.d_model)
    every_weight = []
    for p in params["blocks"]:
        x, weights = block(x, p, cfg)
        if keep_weights:
            every_weight.append(weights)
    x = layer_norm(x, params["final_gain"], params["final_bias"])
    logits = x @ params["embed"].T          # weight tying, chapter 9
    return (logits, every_weight) if keep_weights else logits


def generate(tokens, params, cfg, new_tokens=8, temperature=1.0, seed=0):
    """Chapter 27. Take the last row of logits and sample from it."""
    rng = np.random.default_rng(seed)
    tokens = list(tokens)
    for _ in range(new_tokens):
        window = tokens[-cfg.context:]
        logits = forward(window, params, cfg)[-1]
        probs = softmax(logits / temperature)
        tokens.append(int(rng.choice(len(probs), p=probs)))
    return tokens


def parameter_count(cfg):
    """Chapter 31 derives this formula. Here it is evaluated."""
    per_block = 4 * cfg.d_model ** 2 + 2 * cfg.d_model * cfg.d_ff \
        + cfg.d_ff + cfg.d_model + 4 * cfg.d_model
    return cfg.vocab * cfg.d_model + cfg.n_layers * per_block \
        + 2 * cfg.d_model


def main():
    cfg = Config()
    params = init_params(cfg, seed=0)
    tokens = [3, 14, 7, 1, 22]
    logits, weights = forward(tokens, params, cfg, keep_weights=True)

    print(cfg)
    print("tokens         ", tokens)
    print("logits shape   ", logits.shape)
    print("attention shape", weights[0].shape, "per block")
    print("parameters     ", parameter_count(cfg))

    counted = params["embed"].size + params["final_gain"].size \
        + params["final_bias"].size
    for p in params["blocks"]:
        counted += sum(v.size for v in p.values())
    print("counted        ", counted)

    print("\nlast row of logits, first eight of", cfg.vocab)
    np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
    print(logits[-1][:8])

    print("\nattention of the last query, block 0, head 0")
    print(weights[0][0][-1])

    print("\ngenerated continuation")
    print(generate(tokens, params, cfg, new_tokens=8, seed=1))


if __name__ == "__main__":
    main()
