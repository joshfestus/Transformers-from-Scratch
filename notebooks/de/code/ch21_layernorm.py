"""Kapitel 21. Normierung, und in welche Richtung sie wirkt.

Layer Normalization skaliert jede Position für sich, über die
Modelldimension. Batch Normalization skaliert jedes Merkmal über den
Batch. Der Unterschied ist eine Achse, und er entscheidet, welche sich in
einem Sprachmodell überhaupt benutzen lässt.
"""
import numpy as np
import torch
import torch.nn as nn

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def layer_norm(x, gain=None, bias=None, eps=1e-5):
    """Über die letzte Achse: ein Mittel, eine Varianz pro Position."""
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(variance + eps)
    if gain is not None:
        out = out * gain
    if bias is not None:
        out = out + bias
    return out


def batch_norm(x, eps=1e-5):
    """Über die erste Achse: ein Mittel, eine Varianz pro Merkmal."""
    mean = x.mean(axis=0, keepdims=True)
    variance = x.var(axis=0, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps)


def rms_norm(x, eps=1e-5):
    """Durch die typische Größe teilen, den Mittelwert nicht abziehen."""
    scale = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / scale


class Stack(nn.Module):
    """Ein Residualturm, wahlweise vor jeder Schicht normiert."""

    def __init__(self, depth, width, normalise, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.ModuleList(
            nn.Linear(width, width) for _ in range(depth))
        self.norms = nn.ModuleList(
            nn.LayerNorm(width) for _ in range(depth))
        self.normalise = normalise

    def forward(self, x):
        sizes = []
        for layer, norm in zip(self.layers, self.norms):
            inner = norm(x) if self.normalise else x
            x = x + torch.relu(layer(inner))
            sizes.append(float(x.abs().mean()))
        return x, sizes


def growth(depth=40, width=64, normalise=True, seed=0):
    torch.manual_seed(seed)
    model = Stack(depth, width, normalise, seed)
    with torch.no_grad():
        _, sizes = model(torch.randn(16, width))
    return sizes


def main():
    x = np.array([[10.0, 12.0, 8.0, 14.0],
                  [1.0, 1.5, 0.5, 2.0],
                  [-4.0, 0.0, 4.0, 8.0]])
    print("DREI POSITIONEN, VIER MERKMALE, SEHR VERSCHIEDENE GRÖSSEN")
    print(x)
    print("  Mittelwert je Zeile", x.mean(axis=1))
    print("  Größe je Zeile     ", x.std(axis=1))

    print("\nLAYER NORM ARBEITET ÜBER JEDE ZEILE")
    normed = layer_norm(x)
    print(normed)
    print("  Zeilenmittel", normed.mean(axis=1))
    print("  Zeilengröße ", normed.std(axis=1))

    print("\nBATCH NORM ARBEITET JEDE SPALTE HINUNTER")
    other = batch_norm(x)
    print(other)
    print("  Spaltenmittel", other.mean(axis=0))

    print("\nWARUM EIN SPRACHMODELL DIE ZWEITE NICHT BENUTZEN KANN")
    print("  Batch Norm mischt Beispiele, also hängen die Statistiken")
    print("  eines Satzes davon ab, welche Sätze im selben Batch sind,")
    print("  und beim Generieren gibt es nur eine einzige Sequenz")

    print("\nRMSNORM LÄSST DIE SUBTRAKTION WEG")
    print(rms_norm(x))
    print("  die Zeilen haben nicht mehr den Mittelwert null, und in der")
    print("  Praxis schneiden Modelle damit genauso gut ab")

    print("\nEIN RESIDUALTURM, MIT UND OHNE NORMIERUNG")
    print("  Schicht normiert     nicht normiert")
    plain = growth(40, normalise=False)
    pre = growth(40, normalise=True)
    for i in [0, 9, 19, 29, 39]:
        print(f"  {i:<7} {pre[i]:<12.3f} {plain[i]:.3e}")
    print("  Kapitel 20 ließ die Werte anwachsen. Wer die Eingabe")
    print("  jeder Schicht normiert, macht aus exponentiellem Wachstum")
    print("  lineares, und damit kann ein tiefer Stapel leben")


if __name__ == "__main__":
    main()
