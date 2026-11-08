"""Kapitel 7. Gradientenabstieg, Lernraten und Optimierer.

Eine Schüssel, die in einer Richtung viel steiler ist als in der anderen,
und genau in dieser Lage ist jede echte Loss-Landschaft. Reiner Abstieg
muss in der flachen Richtung kriechen oder in der steilen springen.
Diese Datei misst, was jeder Optimierer dagegen tut.
"""
import numpy as np

STEEP = 200.0           # Krümmung in der zweiten Richtung
START = np.array([-2.4, 0.9])


def loss(point):
    """Quadratische Schüssel, in der zweiten Achse 200-mal steiler."""
    x, y = point
    return 0.5 * (x ** 2 + STEEP * y ** 2)


def gradient(point):
    x, y = point
    return np.array([x, STEEP * y])


def descend(rate, steps=40, start=START):
    """Reiner Gradientenabstieg. Den skalierten Gradienten abziehen."""
    point = np.array(start, dtype=float)
    path = [point.copy()]
    for _ in range(steps):
        point = point - rate * gradient(point)
        path.append(point.copy())
    return np.array(path)


def with_momentum(rate, beta=0.9, steps=40, start=START):
    """Gleitender Mittelwert der Gradienten, und diesem folgen."""
    point = np.array(start, dtype=float)
    velocity = np.zeros_like(point)
    path = [point.copy()]
    for _ in range(steps):
        velocity = beta * velocity + (1 - beta) * gradient(point)
        point = point - rate * velocity
        path.append(point.copy())
    return np.array(path)


def adam(rate, beta1=0.9, beta2=0.999, eps=1e-8, steps=40, start=START):
    """Mittleren Gradienten und typische Größe führen, dann teilen."""
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
    """Lineares Warmup, dann ein Kosinus-Abfall auf null."""
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return peak * 0.5 * (1 + np.cos(np.pi * progress))


def main():
    print("REINER ABSTIEG BEI DREI LERNRATEN")
    print("  Rate     Loss nach 40 Schritten  was passiert")
    for rate, note in [(0.002, "kriecht in der flachen Richtung"),
                       (0.0094, "das Beste, was diese Methode schafft"),
                       (0.0101, "springt, und das Springen wächst")]:
        end = loss(descend(rate)[-1])
        shown = f"{end:.4f}" if end < 1e6 else f"{end:.1e}"
        print(f"  {rate:<8} {shown:<23} {note}")

    print("\nDIE GRÖSSTE RATE, DIE NICHT DIVERGIERT")
    print("  hier ist das zwei durch die steilste Krümmung,")
    print(f"  also 2 / {STEEP:.0f} = {2 / STEEP:.2f}")

    print("\nDREI OPTIMIERER, JEDER BEI SEINER BESTEN RATE, 40 SCHRITTE")
    runs = [("reiner Abstieg", descend(0.0094)),
            ("Momentum", with_momentum(0.060)),
            ("Adam", adam(0.31))]
    print("  Optimierer       Loss am Ende Abstand vom Boden")
    for name, path in runs:
        end = path[-1]
        away = np.linalg.norm(end)
        print(f"  {name:<16} {loss(end):<12.5f} {away:.5f}")

    print("\nEIN ZEITPLAN FÜR DIE LERNRATE")
    print("  Schritt Rate")
    for step in [0, 25, 50, 100, 250, 500, 750, 1000]:
        print(f"  {step:<7} {schedule(step):.3f}")


if __name__ == "__main__":
    main()
