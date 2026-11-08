"""Kapitel 29. Dasselbe Modell in PyTorch, und der Beweis dafür.

Jedes Modul unten ersetzt eine Funktion aus transformer_numpy.py. Die
Gewichte werden hinüberkopiert statt neu initialisiert, also sind die
beiden Modelle nicht bloß ähnlich, sie sind dasselbe Modell, zweimal
ausgedrückt. Die Assertion ganz unten ist der Zweck des Kapitels.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_numpy import (Config, forward as np_forward,
                               init_params, parameter_count,
                               positional_encoding)

torch.set_printoptions(precision=3, sci_mode=False)


class Block(nn.Module):
    """Ein Pre-Norm-Block: Attention, dann ein Feedforward-Netz."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model)

    def heads(self, x):
        n, _ = x.shape
        wide = x.view(n, self.cfg.n_heads, self.cfg.d_head)
        return wide.transpose(0, 1)

    def forward(self, x):
        h = self.ln1(x)
        q, k, v = self.heads(self.q(h)), self.heads(self.k(h)), \
            self.heads(self.v(h))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(0, 1).reshape(x.shape)
        x = x + self.o(attended)
        h = self.ln2(x)
        return x + self.down(F.gelu(self.up(h), approximate="tanh"))


class Transformer(nn.Module):
    """Das ganze Modell. Vergleich es mit der NumPy-Version."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(
            Block(cfg) for _ in range(cfg.n_layers))
        self.final = nn.LayerNorm(cfg.d_model)
        pos = torch.tensor(positional_encoding(cfg.context, cfg.d_model))
        self.register_buffer("pos", pos)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pos[:len(tokens)]
        for block in self.blocks:
            x = block(x)
        return self.final(x) @ self.embed.weight.T


def load_numpy_weights(model, params):
    """Die NumPy-Gewichte hineinkopieren. nn.Linear speichert seine
    Matrix transponiert, weil es x mal W transponiert plus b rechnet."""
    with torch.no_grad():
        model.embed.weight.copy_(torch.tensor(params["embed"]))
        model.final.weight.copy_(torch.tensor(params["final_gain"]))
        model.final.bias.copy_(torch.tensor(params["final_bias"]))
        for block, p in zip(model.blocks, params["blocks"]):
            block.ln1.weight.copy_(torch.tensor(p["ln1_gain"]))
            block.ln1.bias.copy_(torch.tensor(p["ln1_bias"]))
            block.ln2.weight.copy_(torch.tensor(p["ln2_gain"]))
            block.ln2.bias.copy_(torch.tensor(p["ln2_bias"]))
            for layer, key in [(block.q, "w_q"), (block.k, "w_k"),
                               (block.v, "w_v"), (block.o, "w_o"),
                               (block.up, "w_in"), (block.down, "w_out")]:
                layer.weight.copy_(torch.tensor(p[key].T))
            block.up.bias.copy_(torch.tensor(p["b_in"]))
            block.down.bias.copy_(torch.tensor(p["b_out"]))
    return model


def main():
    cfg = Config()
    params = init_params(cfg, seed=0)
    tokens = [3, 14, 7, 1, 22]

    model = Transformer(cfg).double()
    load_numpy_weights(model, params)
    model.eval()

    with torch.no_grad():
        torch_logits = model(torch.tensor(tokens))
    numpy_logits = np_forward(tokens, params, cfg)

    print("Parameter PyTorch", sum(p.numel() for p in model.parameters()))
    print("Parameter NumPy  ", parameter_count(cfg))
    print("Form der Logits  ", tuple(torch_logits.shape))

    print("\nletzte Zeile der Logits, die ersten acht, aus PyTorch")
    print(torch_logits[-1][:8])

    gap = float(np.abs(torch_logits.numpy() - numpy_logits).max())
    print(f"\ngrößte Abweichung vom NumPy-Modell: {gap:.2e}")
    assert gap < 1e-12, "die beiden Modelle sind auseinandergelaufen"
    print("die beiden Implementierungen sind dasselbe Modell")


if __name__ == "__main__":
    main()
