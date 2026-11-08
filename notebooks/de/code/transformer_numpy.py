"""Kapitel 28. Ein vollständiger Transformer, aus den Kapiteln davor.

Ein reines Decoder-Modell: Token-Embeddings, Sinus-Positionen, ein Stapel
von Pre-Norm-Blöcken mit kausaler Multi-Head Attention und einem
Feedforward-Netz, eine abschließende Normierung und ein Ausgabe-Head
zurück zum Vokabular. Jede Funktion hier kam in einem früheren Kapitel
vor, und nichts darin importiert ein Framework.

Das Ganze ist ein Vorwärtsdurchlauf und etwa hundert Zeilen lang.
"""
import numpy as np


class Config:
    """Die Form des Modells, in einem Objekt."""

    def __init__(self, vocab=32, d_model=24, n_heads=4, n_layers=2,
                 d_ff=96, context=16):
        self.vocab = vocab
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.context = context
        assert d_model % n_heads == 0, "Heads müssen die Breite teilen"
        self.d_head = d_model // n_heads

    def __repr__(self):
        return (f"Config(vocab={self.vocab}, d_model={self.d_model}, "
                f"n_heads={self.n_heads}, n_layers={self.n_layers}, "
                f"d_ff={self.d_ff})")


def init_params(cfg, seed=0):
    """Kleine zufällige Gewichte, ein Dictionary pro Block. Kapitel 7
    erklärt, warum die Größe der Anfangswerte zählt."""
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
    """Die Sinusleiter aus Kapitel 10. Nichts zu lernen."""
    position = np.arange(n)[:, None]
    index = np.arange(0, d_model, 2)[None, :]
    rate = np.exp(-np.log(10000.0) * index / d_model)
    encoding = np.zeros((n, d_model))
    encoding[:, 0::2] = np.sin(position * rate)
    encoding[:, 1::2] = np.cos(position * rate)
    return encoding


def layer_norm(x, gain, bias, eps=1e-5):
    """Kapitel 21. Über die Modelldimension normieren, dann skalieren."""
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * gain + bias


def gelu(x):
    """Kapitel 4, in der tanh-Form, die jede Implementierung benutzt."""
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def softmax(x):
    """Kapitel 5. Ums Zeilenmaximum verschieben: kein Überlauf."""
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(n):
    """Kapitel 16. True, wo eine Query hinsehen darf."""
    return np.tril(np.ones((n, n), dtype=bool))


def split_heads(x, n_heads):
    """Aus (n, d_model) wird (n_heads, n, d_head). Kapitel 15."""
    n, d_model = x.shape
    return x.reshape(n, n_heads, d_model // n_heads).transpose(1, 0, 2)


def merge_heads(x):
    """Aus (n_heads, n, d_head) wird (n, d_model). Kapitel 15."""
    n_heads, n, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(n, n_heads * d_head)


def attention(x, p, cfg):
    """Kausale Multi-Head Attention. Kapitel 13 bis 16."""
    q = split_heads(x @ p["w_q"], cfg.n_heads)
    k = split_heads(x @ p["w_k"], cfg.n_heads)
    v = split_heads(x @ p["w_v"], cfg.n_heads)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(cfg.d_head)
    scores = np.where(causal_mask(x.shape[0]), scores, -np.inf)
    weights = softmax(scores)
    return merge_heads(weights @ v) @ p["w_o"], weights


def feedforward(x, p):
    """Kapitel 19. Verbreitern, knicken, verengen."""
    return gelu(x @ p["w_in"] + p["b_in"]) @ p["w_out"] + p["b_out"]


def block(x, p, cfg):
    """Ein Pre-Norm-Block. Kapitel 20, 21 und 23."""
    attended, weights = attention(layer_norm(x, p["ln1_gain"],
                                             p["ln1_bias"]), p, cfg)
    x = x + attended
    x = x + feedforward(layer_norm(x, p["ln2_gain"], p["ln2_bias"]), p)
    return x, weights


def forward(tokens, params, cfg, keep_weights=False):
    """Token-Nummern hinein, eine Zeile Logits pro Position heraus."""
    assert len(tokens) <= cfg.context, "Sequenz länger als der Kontext"
    x = params["embed"][tokens]
    x = x + positional_encoding(len(tokens), cfg.d_model)
    every_weight = []
    for p in params["blocks"]:
        x, weights = block(x, p, cfg)
        if keep_weights:
            every_weight.append(weights)
    x = layer_norm(x, params["final_gain"], params["final_bias"])
    logits = x @ params["embed"].T          # Weight Tying, Kapitel 9
    return (logits, every_weight) if keep_weights else logits


def generate(tokens, params, cfg, new_tokens=8, temperature=1.0, seed=0):
    """Kapitel 27. Die letzte Zeile Logits nehmen und daraus ziehen."""
    rng = np.random.default_rng(seed)
    tokens = list(tokens)
    for _ in range(new_tokens):
        window = tokens[-cfg.context:]
        logits = forward(window, params, cfg)[-1]
        probs = softmax(logits / temperature)
        tokens.append(int(rng.choice(len(probs), p=probs)))
    return tokens


def parameter_count(cfg):
    """Kapitel 31 leitet diese Formel her. Hier wird sie ausgewertet."""
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
    print("Tokens         ", tokens)
    print("Form der Logits", logits.shape)
    print("Form Attention ", weights[0].shape, "pro Block")
    print("Parameter      ", parameter_count(cfg))

    counted = params["embed"].size + params["final_gain"].size \
        + params["final_bias"].size
    for p in params["blocks"]:
        counted += sum(v.size for v in p.values())
    print("gezählt        ", counted)

    print("\nletzte Zeile der Logits, die ersten acht von", cfg.vocab)
    np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
    print(logits[-1][:8])

    print("\nAttention der letzten Query, Block 0, Head 0")
    print(weights[0][0][-1])

    print("\nerzeugte Fortsetzung")
    print(generate(tokens, params, cfg, new_tokens=8, seed=1))


if __name__ == "__main__":
    main()
