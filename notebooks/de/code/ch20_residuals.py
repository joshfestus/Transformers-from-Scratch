"""Kapitel 20. Was eine Residualverbindung mit einem Gradienten macht.

Ein tiefer Stapel von Schichten, zweimal gebaut: einmal schlicht und
einmal so, dass die Eingabe jeder Schicht zu ihrer Ausgabe addiert wird.
An beiden wird dasselbe gemessen, nämlich wie viel Gradient jede Tiefe
erreicht.
"""
import numpy as np
import torch
import torch.nn as nn


class Stack(nn.Module):
    """Ein Turm gleicher Schichten, mit oder ohne Umgehung."""

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
    """Einen Batch durchschicken, dann überall den Gradienten messen."""
    torch.manual_seed(seed)
    model = Stack(depth, width, residual, seed)
    x = torch.randn(16, width)
    loss = model(x).pow(2).mean()
    loss.backward()
    return [float(layer.weight.grad.abs().mean())
            for layer in model.layers]


def activation_size(depth=40, width=64, residual=True, seed=0):
    """Wie groß die Werte selbst auf dem Weg nach oben werden."""
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

    print(f"MITTLERER GRADIENT, DER JEDE SCHICHT ERREICHT, TIEFE {depth}")
    print("  Schicht mit Residuen      ohne")
    for i in [0, 4, 9, 19, 29, depth - 1]:
        print(f"  {i:<7} {with_res[i]:<16.3e} {without[i]:.3e}")

    print("\nWIE TIEF DER GRADIENT VON OBEN NACH UNTEN FÄLLT")
    runs = [("mit Residuen", with_res), ("ohne", without)]
    for name, values in runs:
        ratio = values[-1] / max(values[0], 1e-30)
        print(f"  {name:<16} die oberste Schicht sieht {ratio:.1e}-mal")
        print(f"  {'':<16} so viel wie die unterste")

    print("\nWARUM, IN EINER ZEILE ARITHMETIK")
    print("  ohne Residuum wird der Gradient auf dem Rückweg mit der")
    print("  Gewichtsmatrix jeder Schicht multipliziert, die Faktoren")
    print("  schaukeln sich auf. Mit Residuum ist die Ableitung von")
    print("  x + f(x) gleich 1 + f', also gibt es immer einen Rückweg,")
    print("  der mit eins multipliziert")

    print("\nDERSELBE STAPEL, VIEL TIEFER")
    print("  Tiefe   mit Residuen     ohne")
    for d in [10, 40, 100]:
        a = gradient_by_depth(d, residual=True)
        b = gradient_by_depth(d, residual=False)
        print(f"  {d:<7} {a[0]:<16.3e} {b[0]:.3e}")
    print("  der Gradient erreicht jetzt den Boden, und er ist unterwegs")
    print("  riesig geworden. Residuen haben ein Problem gelöst und")
    print("  ein anderes geschaffen")

    print("\nWARUM ER WUCHS, GEMESSEN AUF DEM WEG NACH OBEN")
    print("  Schicht Größe der Aktivierung mit Residuen ohne")
    grew = activation_size(40, residual=True)
    flat = activation_size(40, residual=False)
    for i in [0, 9, 19, 29, 39]:
        print(f"  {i:<7} {grew[i]:<31.3e} {flat[i]:.3e}")
    print("  wer bei jeder Schicht die Eingabe wieder addiert, lässt")
    print("  die Werte selbst anwachsen. Kapitel 21 hält das auf")


if __name__ == "__main__":
    main()
