"""Kapitel 22. Dropout, und der Faktor, der es stimmig macht.

Beim Training werden zufällig einige Aktivierungen auf null gesetzt. Bei
der Inferenz keine. Beides muss im Mittel übereinstimmen, und der
Skalierungsfaktor sorgt dafür.
"""
import numpy as np

rng = np.random.default_rng(0)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def dropout(x, rate, training=True):
    """Inverted Dropout, wie es jede Bibliothek umsetzt."""
    if not training or rate == 0.0:
        return x
    keep = rng.random(x.shape) >= rate
    return x * keep / (1.0 - rate)


def naive_dropout(x, rate, training=True):
    """Die Version ohne Skalierung, zum Vergleich."""
    if not training:
        return x
    return x * (rng.random(x.shape) >= rate)


def main():
    x = np.ones((4, 6))
    print("SECHS AKTIVIERUNGEN, ALLE EINS, DROPOUT-RATE EIN DRITTEL")
    print(dropout(x, 1 / 3)[:2])
    print("  die Überlebenden sind 1.5, nicht 1.0, weil jeder durch den")
    print("  Anteil geteilt wurde, der überlebt")

    print("\nDER MITTELWERT BLEIBT ERHALTEN, UND DARUM GEHT ES")
    big = np.ones((20000, 8))
    for rate in [0.0, 0.1, 0.3, 0.5, 0.9]:
        scaled = dropout(big, rate).mean()
        plain = naive_dropout(big, rate).mean()
        print(f"  Rate {rate:<5} mit Skalierung {scaled:.4f}   "
              f"ohne {plain:.4f}")
    print("  ohne die Skalierung würde das Abschalten des Dropouts bei")
    print("  der Inferenz die Größe jeder Aktivierung ändern")

    print("\nTRAINING UND INFERENZ, DIESELBE EINGABE")
    small = np.arange(1.0, 7.0).reshape(1, 6)
    print("  Eingabe   ", small[0])
    print("  Training  ", dropout(small, 0.5)[0])
    print("  Training  ", dropout(small, 0.5)[0])
    print("  Inferenz  ", dropout(small, 0.5, training=False)[0])
    print("  derselbe Aufruf gibt beim Training jedes Mal eine andere")
    print("  Antwort und bei der Inferenz jedes Mal dieselbe")

    print("\nWIE VIEL VARIANZ DROPOUT HINZUFÜGT")
    print("  Rate    Streuung des Batch-Mittelwerts")
    for rate in [0.0, 0.1, 0.3, 0.5]:
        ones = np.ones((32, 64))
        means = [dropout(ones, rate).mean() for _ in range(400)]
        print(f"  {rate:<7} {np.std(means):.5f}")
    print("  eine höhere Rate regularisiert stärker und rauscht mehr")


if __name__ == "__main__":
    main()
