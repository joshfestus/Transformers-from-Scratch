"""Chapter 20. What a residual connection does to a gradient.

A deep stack of layers, built twice: once plainly and once with the
input of each layer added to its output. The same measurement is taken
on both, which is how much gradient reaches each depth.
"""
import numpy as np
import torch
import torch.nn as nn


class Stack(nn.Module):
    """A tower of identical layers, with or without the bypass."""

    def __init__(self, depth, width, residual, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.ModuleList(
            nn.Linear(width, width) for _ in range(depth))
        self.residual = residual

    def forward(self, x):
        for layer in self.layers:
            out = torch.relu(layer(x))
            x = x + out if self.residual else out
        return x


def gradient_by_depth(depth=40, width=64, residual=True, seed=0):
    """Push one batch through, then measure the gradient everywhere."""
    torch.manual_seed(seed)
    model = Stack(depth, width, residual, seed)
    x = torch.randn(16, width)
    loss = model(x).pow(2).mean()
    loss.backward()
    return [float(layer.weight.grad.abs().mean())
            for layer in model.layers]


def activation_size(depth=40, width=64, residual=True, seed=0):
    """How large the values themselves get on the way up the stack."""
    torch.manual_seed(seed)
    model = Stack(depth, width, residual, seed)
    x = torch.randn(16, width)
    sizes = []
    with torch.no_grad():
        for layer in model.layers:
            out = torch.relu(layer(x))
            x = x + out if residual else out
            sizes.append(float(x.abs().mean()))
    return sizes


def main():
    depth = 40
    with_res = gradient_by_depth(depth, residual=True)
    without = gradient_by_depth(depth, residual=False)

    print(f"AVERAGE GRADIENT REACHING EACH LAYER, DEPTH {depth}")
    print("  layer   with residuals   without")
    for i in [0, 4, 9, 19, 29, depth - 1]:
        print(f"  {i:<7} {with_res[i]:<16.3e} {without[i]:.3e}")

    print("\nHOW FAR THE GRADIENT FALLS FROM TOP TO BOTTOM")
    runs = [("with residuals", with_res), ("without", without)]
    for name, values in runs:
        ratio = values[-1] / max(values[0], 1e-30)
        print(f"  {name:<16} the top layer sees {ratio:.1e} times what")
        print(f"  {'':<16} the bottom layer sees")

    print("\nWHY, IN ONE LINE OF ARITHMETIC")
    print("  without a residual, the gradient is multiplied by each")
    print("  layer's weight matrix on the way back, and those factors")
    print("  compound. With one, the derivative of x + f(x) is 1 + f',")
    print("  so there is always a route back that multiplies by one")

    print("\nTHE SAME STACK, MUCH DEEPER")
    print("  depth   with residuals   without")
    for d in [10, 40, 100]:
        a = gradient_by_depth(d, residual=True)
        b = gradient_by_depth(d, residual=False)
        print(f"  {d:<7} {a[0]:<16.3e} {b[0]:.3e}")
    print("  the gradient now reaches the bottom, and it has grown")
    print("  enormous on the way. Residuals fixed one problem and")
    print("  created another")

    print("\nWHY IT GREW, MEASURED ON THE WAY UP")
    print("  layer   activation size with residuals   without")
    grew = activation_size(40, residual=True)
    flat = activation_size(40, residual=False)
    for i in [0, 9, 19, 29, 39]:
        print(f"  {i:<7} {grew[i]:<31.3e} {flat[i]:.3e}")
    print("  adding the input back at every layer makes the values")
    print("  themselves compound. Chapter 21 is what stops that")


if __name__ == "__main__":
    main()
