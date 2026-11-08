"""Kapitel 10. Die Position, die Information, die Attention wegwirft.

Attention behandelt ihre Eingabe als Menge. Mischt man die Zeilen, kommen
dieselben Zeilen heraus, genauso gemischt. Die Reihenfolge trägt also
keinerlei Information, bis etwas sie hineinbringt. Diese Datei zeigt
zuerst dieses Versagen und dann die beiden üblichen Reparaturen.
"""
import numpy as np

from ch14_attention import W_K, W_Q, W_V, X, attention

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def positional_encoding(n, d_model):
    """Die Sinusleiter. Gerade Spalten Sinus, ungerade Kosinus."""
    position = np.arange(n)[:, None]
    index = np.arange(0, d_model, 2)[None, :]
    rate = np.exp(-np.log(10000.0) * index / d_model)
    encoding = np.zeros((n, d_model))
    encoding[:, 0::2] = np.sin(position * rate)
    encoding[:, 1::2] = np.cos(position * rate)
    return encoding


def rotate(vector, angle):
    """Rotary Position Embedding, an einem einzelnen Dimensionspaar."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vector[0] - s * vector[1],
                     s * vector[0] + c * vector[1]])


def order_blindness():
    """Attention auf einen Satz und auf eine gemischte Kopie davon."""
    order = np.array([3, 0, 2, 1])
    plain = attention(X, W_Q, W_K, W_V)[-1]
    shuffled = attention(X[order], W_Q, W_K, W_V)[-1]
    return order, plain, shuffled


def main():
    order, plain, shuffled = order_blindness()
    print("ATTENTION AUF DEN SATZ, WIE ER DASTEHT")
    print(plain)
    print("\nDIESELBEN VIER ZEILEN, GEMISCHT, DURCH DIESELBE ATTENTION")
    print(shuffled)
    print("\nDIE ZWEITE IST DIE ERSTE, MIT ZEILEN IN NEUER REIHENFOLGE")
    print("  identisch:", np.allclose(shuffled, plain[order]))
    print("  nichts in der Attention weiß also, welches Wort zuerst kam")

    print("\nDIE SINUSKODIERUNG, VIER POSITIONEN MAL SECHS ZAHLEN")
    encoding = positional_encoding(4, 6)
    for i, row in enumerate(encoding):
        print(f"  Position {i}  {row}")

    print("\nJEDES SPALTENPAAR DREHT SICH IN SEINEM EIGENEN TEMPO")
    d_model = 64
    for index in [0, 2, 8, 32]:
        rate = np.exp(-np.log(10000.0) * index / d_model)
        period = 2 * np.pi / rate
        print(f"  Spalten {index} und {index + 1}: eine Drehung alle "
              f"{period:.0f} Positionen")

    print("\nDAS SKALARPRODUKT ZWEIER KODIERUNGEN, JE NACH ABSTAND")
    long_encoding = positional_encoding(40, 64)
    print("  Abst. Skalarprodukt mit Position 10")
    for gap in [0, 1, 2, 5, 10, 20]:
        value = float(long_encoding[10] @ long_encoding[10 + gap])
        print(f"  {gap:<5} {value:.3f}")

    print("\nROTARY-POSITIONEN DREHEN, STATT ZU ADDIEREN")
    vector = np.array([1.0, 0.0])
    for position in range(4):
        turned = rotate(vector, position * np.pi / 6)
        print(f"  Position {position}  {turned}")
    a = rotate(vector, 1 * np.pi / 6)
    b = rotate(vector, 3 * np.pi / 6)
    c = rotate(vector, 5 * np.pi / 6)
    d = rotate(vector, 7 * np.pi / 6)
    print(f"  Skalarprodukt Positionen 1 und 3  {float(a @ b):.3f}")
    print(f"  Skalarprodukt Positionen 5 und 7  {float(c @ d):.3f}")
    print("  derselbe Abstand gibt dieselbe Antwort, egal wo")


if __name__ == "__main__":
    main()
