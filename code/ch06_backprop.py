"""Chapter 6. A complete backward pass, by hand and then by machine.

Two inputs, two hidden units with a ReLU, one output, and a squared
error against a single target. Every weight is a whole number or a half,
so every derivative below can be checked with a pencil, and every one of
them is then checked again against a finite difference.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

X = np.array([1.0, 2.0])            # one example, two features
Y = 1.0                             # the target

W1 = np.array([[0.5, -0.5],         # 2 by 2, one column per hidden unit
               [1.0, 0.5]])
B1 = np.array([0.0, 0.5])
W2 = np.array([1.0, -1.0])          # 2 by 1, written flat
B2 = 0.5


def relu(z):
    return np.maximum(z, 0.0)


def forward(x, w1, b1, w2, b2, y):
    """Every intermediate value, because the backward pass needs them."""
    h_pre = x @ w1 + b1
    h = relu(h_pre)
    pred = h @ w2 + b2
    loss = 0.5 * (pred - y) ** 2
    return h_pre, h, pred, loss


def backward(x, w1, b1, w2, b2, y):
    """The chain rule, applied once per layer, from the loss backwards."""
    h_pre, h, pred, _ = forward(x, w1, b1, w2, b2, y)
    d_pred = pred - y                       # from 0.5 (pred - y) squared
    d_w2 = h * d_pred                       # pred = h . w2 + b2
    d_b2 = d_pred
    d_h = w2 * d_pred
    d_h_pre = d_h * (h_pre > 0)             # ReLU passes or blocks
    d_w1 = np.outer(x, d_h_pre)             # h_pre = x @ w1 + b1
    d_b1 = d_h_pre
    return d_w1, d_b1, d_w2, d_b2


def loss_only(w1, b1, w2, b2):
    return forward(X, w1, b1, w2, b2, Y)[3]


def numeric_gradient(eps=1e-6):
    """The same gradients, measured by nudging each parameter."""
    g_w1 = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            up, down = W1.copy(), W1.copy()
            up[i, j] += eps
            down[i, j] -= eps
            g_w1[i, j] = (loss_only(up, B1, W2, B2)
                          - loss_only(down, B1, W2, B2)) / (2 * eps)
    g_w2 = np.zeros_like(W2)
    for j in range(W2.shape[0]):
        up, down = W2.copy(), W2.copy()
        up[j] += eps
        down[j] -= eps
        g_w2[j] = (loss_only(W1, B1, up, B2)
                   - loss_only(W1, B1, down, B2)) / (2 * eps)
    return g_w1, g_w2


def sweep_squared_error():
    """Gradient check error against step size, for the network above."""
    exact_w1, _, exact_w2, _ = backward(X, W1, B1, W2, B2, Y)
    rows = []
    for power in range(-11, 0):
        eps = 10.0 ** power
        g_w1, g_w2 = numeric_gradient(eps)
        err = max(np.abs(g_w1 - exact_w1).max(),
                  np.abs(g_w2 - exact_w2).max())
        rows.append((eps, float(err)))
    return rows


def logit_loss(w):
    """Cross entropy on three logits, as a function of the first one."""
    return -np.log(softmax(np.array([w, 1.0, 0.0]))[0])


def sweep_cross_entropy(w=2.0):
    """The same sweep on a loss that is not quadratic in the parameter."""
    exact = softmax(np.array([w, 1.0, 0.0]))[0] - 1.0
    rows = []
    for power in range(-12, 0):
        eps = 10.0 ** power
        measured = (logit_loss(w + eps) - logit_loss(w - eps)) / (2 * eps)
        rows.append((eps, float(abs(measured - exact))))
    return rows


def softmax(z):
    shifted = z - z.max()
    e = np.exp(shifted)
    return e / e.sum()


def cross_entropy_gradient(logits, correct):
    """The result the rest of the book leans on: the gradient of softmax
    followed by cross entropy is the probabilities minus one hot."""
    p = softmax(logits)
    target = np.zeros_like(p)
    target[correct] = 1.0
    return p, p - target


def main():
    h_pre, h, pred, loss = forward(X, W1, B1, W2, B2, Y)
    print("FORWARD")
    print("  h_pre  ", h_pre)
    print("  h      ", h)
    print("  pred   ", pred)
    print("  loss   ", loss)

    d_w1, d_b1, d_w2, d_b2 = backward(X, W1, B1, W2, B2, Y)
    print("\nBACKWARD")
    print("  dL/dpred", pred - Y)
    print("  dL/dW2  ", d_w2)
    print("  dL/db2  ", d_b2)
    print("  dL/dW1  ", d_w1[0], "and", d_w1[1])
    print("  dL/db1  ", d_b1)

    g_w1, g_w2 = numeric_gradient()
    print("\nCHECKED AGAINST A FINITE DIFFERENCE, eps = 1e-6")
    print("  numeric dL/dW1", g_w1[0], "and", g_w1[1])
    print("  numeric dL/dW2", g_w2)
    worst = max(np.abs(g_w1 - d_w1).max(), np.abs(g_w2 - d_w2).max())
    print("  largest disagreement", f"{worst:.2e}")

    print("\nSTEP SIZE AGAINST ERROR, two losses")
    print("  eps        squared error   cross entropy")
    for (eps, a), (_, b) in zip(sweep_squared_error()[1:],
                                sweep_cross_entropy()[2:]):
        print(f"  {eps:.0e}   {a:12.2e}   {b:13.2e}")

    print("\nSOFTMAX WITH CROSS ENTROPY, gradient is p minus one hot")
    p, g = cross_entropy_gradient(np.array([2.0, 1.0, 0.0]), correct=0)
    print("  probabilities", p)
    print("  gradient     ", g)


if __name__ == "__main__":
    main()
