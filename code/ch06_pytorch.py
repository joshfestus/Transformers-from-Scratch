"""Chapter 6, section 7. The same gradients from autograd.

Nothing here is new arithmetic. PyTorch records the operations as they
run, then walks that record backwards applying the same local
derivatives written out by hand in ch06_backprop.py.
"""
import numpy as np
import torch

from ch06_backprop import B1, B2, W1, W2, X, Y, backward

torch.set_printoptions(precision=4)


def main():
    x = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(Y, dtype=torch.float64)
    w1 = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(B1, dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(B2, dtype=torch.float64, requires_grad=True)

    h = torch.relu(x @ w1 + b1)
    pred = h @ w2 + b2
    loss = 0.5 * (pred - y) ** 2
    loss.backward()

    print("loss    ", float(loss))
    print("W1.grad ", w1.grad[0].tolist(), "and", w1.grad[1].tolist())
    print("b1.grad ", b1.grad.tolist())
    print("W2.grad ", w2.grad.tolist())
    print("b2.grad ", float(b2.grad))

    d_w1, d_b1, d_w2, d_b2 = backward(X, W1, B1, W2, B2, Y)
    assert np.allclose(w1.grad.numpy(), d_w1)
    assert np.allclose(b1.grad.numpy(), d_b1)
    assert np.allclose(w2.grad.numpy(), d_w2)
    assert np.allclose(float(b2.grad), d_b2)
    print("\nautograd agrees with the gradients worked out by hand")


if __name__ == "__main__":
    main()
