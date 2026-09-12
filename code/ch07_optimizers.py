"""Chapter 7. Gradient descent, learning rates and optimizers.

A bowl that is much steeper in one direction than the other, which is
the situation every real loss surface is in. Plain descent has to creep
along the shallow direction or bounce along the steep one. This file
measures what each optimizer does about that.
"""
import numpy as np

STEEP = 200.0           # curvature in the second direction
START = np.array([-2.4, 0.9])


def loss(point):
    """Half a quadratic bowl, twenty times steeper in the second axis."""
    x, y = point
    return 0.5 * (x ** 2 + STEEP * y ** 2)


def gradient(point):
    x, y = point
    return np.array([x, STEEP * y])


def descend(rate, steps=40, start=START):
    """Plain gradient descent. Subtract the gradient, scaled."""
    point = np.array(start, dtype=float)
    path = [point.copy()]
    for _ in range(steps):
        point = point - rate * gradient(point)
        path.append(point.copy())
    return np.array(path)


def with_momentum(rate, beta=0.9, steps=40, start=START):
    """Keep a running average of past gradients and step along that."""
    point = np.array(start, dtype=float)
    velocity = np.zeros_like(point)
    path = [point.copy()]
    for _ in range(steps):
        velocity = beta * velocity + (1 - beta) * gradient(point)
        point = point - rate * velocity
        path.append(point.copy())
    return np.array(path)


def adam(rate, beta1=0.9, beta2=0.999, eps=1e-8, steps=40, start=START):
    """Track the average gradient and its typical size, then divide."""
    point = np.array(start, dtype=float)
    m = np.zeros_like(point)
    v = np.zeros_like(point)
    path = [point.copy()]
    for step in range(1, steps + 1):
        g = gradient(point)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        point = point - rate * m_hat / (np.sqrt(v_hat) + eps)
        path.append(point.copy())
    return np.array(path)


def schedule(step, total=1000, warmup=100, peak=1.0):
    """Linear warmup, then a cosine decay to nought."""
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return peak * 0.5 * (1 + np.cos(np.pi * progress))


def main():
    print("PLAIN DESCENT AT THREE LEARNING RATES")
    print("  rate     loss after 40 steps   what happened")
    for rate, note in [(0.002, "creeps along the shallow direction"),
                       (0.0094, "the best this method can manage"),
                       (0.0101, "bounces, and the bounce grows")]:
        end = loss(descend(rate)[-1])
        shown = f"{end:.4f}" if end < 1e6 else f"{end:.1e}"
        print(f"  {rate:<8} {shown:<21} {note}")

    print("\nTHE LARGEST RATE THAT DOES NOT DIVERGE")
    print("  for this bowl it is two divided by the steepest curvature,")
    print(f"  which is 2 / {STEEP:.0f} = {2 / STEEP:.2f}")

    print("\nTHREE OPTIMIZERS, EACH AT ITS OWN BEST RATE, 40 STEPS")
    runs = [("plain descent", descend(0.0094)),
            ("momentum", with_momentum(0.060)),
            ("adam", adam(0.31))]
    print("  optimizer        final loss   distance from the bottom")
    for name, path in runs:
        end = path[-1]
        away = np.linalg.norm(end)
        print(f"  {name:<16} {loss(end):<12.5f} {away:.5f}")

    print("\nA LEARNING RATE SCHEDULE")
    print("  step    rate")
    for step in [0, 25, 50, 100, 250, 500, 750, 1000]:
        print(f"  {step:<7} {schedule(step):.3f}")


if __name__ == "__main__":
    main()
