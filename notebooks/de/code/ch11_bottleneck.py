"""Kapitel 11. Was eine Zusammenfassung fester Größe fassen kann.

Vor der Attention presste ein Sequenzmodell alles, was es gelesen hatte,
in einen einzigen Vektor fester Breite und reichte ihn weiter. Diese
Datei misst, wie viel diese Kompression übersteht, und was stattdessen
passiert, wenn nichts weggeworfen wird.
"""
import numpy as np

rng = np.random.default_rng(0)


def unit_vectors(count, width):
    """Zufällige Richtungen, jede mit der Länge eins."""
    v = rng.standard_normal((count, width))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def summary_test(length, width, trials=4000):
    """Alle Vektoren aufaddieren und fragen, was die Summe noch weiß.

    Gefragt wird, ob ein Vektor unter den addierten war. Eine Probe wird
    nach ihrem Skalarprodukt mit der Summe bewertet: Etwas Addiertes kommt
    auf etwa eins plus Störung durch den Rest, etwas nicht Addiertes auf
    etwa null plus dieselbe Störung. Geprüft wird, ob sich die beiden
    noch auseinanderhalten lassen.
    """
    hits = 0
    for _ in range(trials):
        items = unit_vectors(length + 1, width)
        summary = items[:length].sum(axis=0)
        stored = items[rng.integers(length)] @ summary
        novel = items[length] @ summary
        hits += int(stored > novel)
    return hits / trials


def attention_test(length, width, trials=4000):
    """Dieselbe Frage, aber jeder Vektor wird behalten statt addiert."""
    hits = 0
    for _ in range(trials):
        items = unit_vectors(length + 1, width)
        kept = items[:length]
        stored = (kept @ items[rng.integers(length)]).max()
        novel = (kept @ items[length]).max()
        hits += int(stored > novel)
    return hits / trials


def noise_growth(length, width):
    """Die Größe der Störung im Verhältnis zum Signal, theoretisch."""
    return np.sqrt((length - 1) / width)


def main():
    width = 32
    print(f"IST DIESER VEKTOR DARIN? ZUSAMMENFASSUNG DER BREITE {width}")
    print("  Anzahl  aus der Summe      alle behalten      Störung")
    for length in [2, 4, 8, 16, 32, 64, 128]:
        summed = summary_test(length, width)
        kept = attention_test(length, width)
        print(f"  {length:<7} {summed:<18.3f} {kept:<18.3f} "
              f"{noise_growth(length, width):.2f}")

    print("\nEINE BREITERE ZUSAMMENFASSUNG FASST ANTEILIG MEHR")
    print("  Breite  Anzahl  richtig beantwortet  Störung")
    for w in [32, 128, 512, 2048]:
        print(f"  {w:<7} {64:<7} {summary_test(64, w):<20.3f} "
              f"{noise_growth(64, w):.2f}")

    print("\nWAS EINE SEQUENZ KOSTET, IN NICHT ÜBERLAPPENDEN SCHRITTEN")
    print("  Länge    rekurrente Schritte  Attention-Schritte")
    for length in [10, 100, 1000]:
        print(f"  {length:<8} {length:<20} {1}")


if __name__ == "__main__":
    main()
