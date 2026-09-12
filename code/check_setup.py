"""Appendix D. Three lines that confirm the setup works."""
import numpy as np
import torch


def main():
    print("numpy", np.__version__)
    print("torch", torch.__version__)
    x = torch.randn(3, 4)
    print("a tensor", tuple(x.shape), "sums to", round(float(x.sum()), 3))


if __name__ == "__main__":
    main()
