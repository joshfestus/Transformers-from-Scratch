"""Anhang D. Drei Zeilen, die bestätigen, dass alles funktioniert."""
import numpy as np
import torch


def main():
    print("numpy", np.__version__)
    print("torch", torch.__version__)
    x = torch.randn(3, 4)
    print("ein Tensor", tuple(x.shape), "mit der Summe",
          round(float(x.sum()), 3))


if __name__ == "__main__":
    main()
