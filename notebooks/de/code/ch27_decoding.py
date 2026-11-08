"""Kapitel 27. Aus einer Zeile Scores wird ein Wort, auf fünf Arten.

Das Modell erzeugt einen Score pro Token. Jede Strategie in dieser Datei
ist eine andere Antwort auf dieselbe Frage: Welches nehmen wir?
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

WORDS = ["Milch", "Wasser", "Tee", "Kaffee", "Saft", "Bier", "Suppe",
         "Tinte"]
LOGITS = np.array([3.2, 2.6, 1.9, 1.5, 0.9, 0.4, -0.6, -2.0])


def softmax(scores, temperature=1.0):
    scaled = np.asarray(scores, dtype=float) / temperature
    shifted = scaled - scaled.max()
    e = np.exp(shifted)
    return e / e.sum()


def greedy(logits):
    return int(np.argmax(logits))


def top_k_filter(probs, k):
    """Die k größten behalten, neu normieren, den Rest verwerfen."""
    kept = np.zeros_like(probs)
    order = np.argsort(probs)[::-1][:k]
    kept[order] = probs[order]
    return kept / kept.sum()


def nucleus_filter(probs, p):
    """Die kleinste Gruppe behalten, deren Summe p erreicht."""
    order = np.argsort(probs)[::-1]
    running = np.cumsum(probs[order])
    cut = int(np.searchsorted(running, p)) + 1
    kept = np.zeros_like(probs)
    kept[order[:cut]] = probs[order[:cut]]
    return kept / kept.sum()


def repetition_penalty(logits, used, strength=1.2):
    """Den Score von allem drücken, was schon erzeugt wurde."""
    out = np.array(logits, dtype=float)
    for token in used:
        out[token] = (out[token] / strength if out[token] > 0
                      else out[token] * strength)
    return out


def main():
    probs = softmax(LOGITS)
    print("EINE ZEILE SCORES, UND DIE WAHRSCHEINLICHKEITEN DARAUS")
    for word, score, p in zip(WORDS, LOGITS, probs):
        print(f"  {word:<8} Score {score:>5.1f}   Wahrscheinl. {p:.4f}")

    print("\nGREEDY NIMMT JEDES MAL DAS GRÖSSTE")
    print(f"  {WORDS[greedy(LOGITS)]}, mit Wahrscheinlichkeit "
          f"{probs.max():.4f}")
    print("  dieselbe Eingabe gibt immer dieselbe Ausgabe")

    print("\nDIE TEMPERATUR ÄNDERT, WIE ENTSCHIEDEN DIE WAHL IST")
    print("  T      erstes Wort Anteil     Wörter über 1 Prozent")
    for t in [0.25, 0.7, 1.0, 1.5, 3.0]:
        scaled = softmax(LOGITS, t)
        print(f"  {t:<6} {WORDS[int(scaled.argmax())]:<11} "
              f"{scaled.max():<11.4f} {int((scaled > 0.01).sum())}")

    print("\nTOP-K BEHÄLT EINE FESTE ANZAHL")
    for k in [1, 3, 5]:
        kept = top_k_filter(probs, k)
        alive = [w for w, v in zip(WORDS, kept) if v > 0]
        print(f"  k={k}  {', '.join(alive)}")

    print("\nNUCLEUS BEHÄLT SO VIELE, WIE ES FÜR EINEN ANTEIL BRAUCHT")
    for p in [0.5, 0.8, 0.95]:
        kept = nucleus_filter(probs, p)
        alive = [w for w, v in zip(WORDS, kept) if v > 0]
        print(f"  p={p}  {len(alive)} Wörter: {', '.join(alive)}")
    print("  eine flache Verteilung behält viele, eine scharfe wenige,")
    print("  und genau deshalb wird es Top-k vorgezogen")

    print("\nWAS EINE STRAFE FÜR WIEDERHOLUNG TUT")
    used = [0, 1]
    penalised = softmax(repetition_penalty(LOGITS, used))
    print("  Wort      vorher   nachher")
    for i, word in enumerate(WORDS[:4]):
        print(f"  {word:<9} {probs[i]:.4f}   {penalised[i]:.4f}")

    print("\nDIESELBE ZEILE ZWANZIGMAL ZIEHEN")
    rng = np.random.default_rng(0)
    for name, filtered in [("ohne Filter", probs),
                           ("Top-k 3", top_k_filter(probs, 3)),
                           ("Nucleus 0.9", nucleus_filter(probs, 0.9))]:
        draws = rng.choice(len(WORDS), size=20, p=filtered)
        counts = {WORDS[i]: int((draws == i).sum()) for i in set(draws)}
        shown = ", ".join(f"{w} {c}" for w, c in sorted(counts.items()))
        print(f"  {name:<13} {shown}")


if __name__ == "__main__":
    main()
