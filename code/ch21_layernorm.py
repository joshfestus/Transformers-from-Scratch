"""Chapter 21. Normalisation, and which direction it works in.

Layer normalisation rescales each position on its own, across the model
dimension. Batch normalisation rescales each feature across the batch.
The difference is one axis and it decides which one can be used in a
language model at all.
"""
import numpy as np
import torch
import torch.nn as nn

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def layer_norm(x, gain=None, bias=None, eps=1e-5):
    """Across the last axis: one mean and one variance per position."""
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(variance + eps)
    if gain is not None:
        out = out * gain
    if bias is not None:
        out = out + bias
    return out


def batch_norm(x, eps=1e-5):
    """Across the first axis: one mean and one variance per feature."""
    mean = x.mean(axis=0, keepdims=True)
    variance = x.var(axis=0, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps)


def rms_norm(x, eps=1e-5):
    """Divide by the typical size, and do not subtract the mean."""
    scale = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / scale


class Stack(nn.Module):
    """A residual tower, optionally normalising before each layer."""

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
    print("THREE POSITIONS, FOUR FEATURES, VERY DIFFERENT SCALES")
    print(x)
    print("  each row's mean", x.mean(axis=1))
    print("  each row's size", x.std(axis=1))

    print("\nLAYER NORM WORKS ACROSS EACH ROW")
    normed = layer_norm(x)
    print(normed)
    print("  row means", normed.mean(axis=1))
    print("  row sizes", normed.std(axis=1))

    print("\nBATCH NORM WORKS DOWN EACH COLUMN")
    other = batch_norm(x)
    print(other)
    print("  column means", other.mean(axis=0))

    print("\nWHY A LANGUAGE MODEL CANNOT USE THE SECOND")
    print("  batch norm mixes examples together, so one sentence's")
    print("  statistics depend on which other sentences share its batch,")
    print("  and at generation time there is only one sequence")

    print("\nRMSNORM DROPS THE SUBTRACTION")
    print(rms_norm(x))
    print("  the rows no longer have mean nought, and in practice")
    print("  models trained with it do just as well")

    print("\nA RESIDUAL TOWER, WITH AND WITHOUT NORMALISATION")
    print("  layer   normalised   not normalised")
    plain = growth(40, normalise=False)
    pre = growth(40, normalise=True)
    for i in [0, 9, 19, 29, 39]:
        print(f"  {i:<7} {pre[i]:<12.3f} {plain[i]:.3e}")
    print("  chapter 20 left the values multiplying up. Normalising the")
    print("  input to each layer turns that growth from exponential into")
    print("  linear, which a deep stack can live with")


if __name__ == "__main__":
    main()
