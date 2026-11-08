"""Kapitel 16. Eine Position daran hindern, zu sehen, was sie nicht soll.

Zwei Masken. Eine kausale Maske hält eine Position davon ab, die Zukunft
zu lesen, und erst das macht Generieren möglich. Eine Padding-Maske hält
echte Positionen davon ab, auf das Füllmaterial zu achten, mit dem ein
Batch rechteckig gemacht wird. Beide werden vor der Softmax angewendet,
und diese Datei zeigt, was passiert, wenn man sie danach anwendet.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

SCORES = np.array([[2.0, 1.0, 0.5, 0.0],
                   [1.0, 3.0, 1.5, 0.5],
                   [0.5, 1.0, 2.0, 1.0],
                   [0.0, 0.5, 1.5, 2.5]])


def softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(n):
    """True, wo eine Query in Zeile i einen Key in Spalte j lesen darf."""
    return np.tril(np.ones((n, n), dtype=bool))


def padding_mask(lengths, n):
    """True für echte Positionen, False für das Füllmaterial danach."""
    return np.arange(n)[None, :] < np.asarray(lengths)[:, None]


def masked_before(scores, allowed):
    """Der richtige Weg. Verbotene Scores auf minus unendlich setzen."""
    return softmax(np.where(allowed, scores, -np.inf))


def masked_after(scores, allowed):
    """Der falsche Weg. Erst normieren, dann Verbotenes auf null."""
    return np.where(allowed, softmax(scores), 0.0)


def main():
    n = 4
    allowed = causal_mask(n)
    print("DIE KAUSALE MASKE, TRUE HEISST ERLAUBT")
    for row in allowed:
        print("  " + "  ".join(" ja" if v else " . " for v in row))

    print("\nSCORES VOR JEDER MASKE")
    print(SCORES)

    print("\nVOR DER SOFTMAX MASKIERT, UND DAS IST RICHTIG")
    right = masked_before(SCORES, allowed)
    print(right)
    print("  Zeilensummen", right.sum(axis=1))

    print("\nNACH DER SOFTMAX MASKIERT, UND DAS IST FALSCH")
    wrong = masked_after(SCORES, allowed)
    print(wrong)
    sums = wrong.sum(axis=1)
    print("  Zeilensummen", sums)
    print(f"  die erste Zeile behält nur {sums[0]:.3f} ihres Gewichts,")
    print("  und der Rest ist nirgendwohin gegangen")

    print("\nWIE WEIT DIE BEIDEN AUSEINANDERLIEGEN")
    print(f"  größter Unterschied {np.abs(right - wrong).max():.3f}")

    print("\nEINE PADDING-MASKE FÜR EINEN BATCH AUS DREI")
    lengths = [4, 2, 3]
    pad = padding_mask(lengths, n)
    for length, row in zip(lengths, pad):
        marks = "  ".join(" ja" if v else " . " for v in row)
        print(f"  Länge {length}   {marks}")

    print("\nEINE MASKENKONSTANTE, DIE NICHT NEGATIV GENUG IST")
    for constant in [-6.0, -20.0, -1e4]:
        leaked = softmax(np.where(allowed, SCORES, constant))[0, 1:].sum()
        print(f"  maskiert mit {constant:>9}  verbotenes Restgewicht "
              f"{leaked:.3e}")
    print("  eine mäßige Konstante lässt etwas durch. Eine große läuft")
    print("  bei 64 Bit nach unten über, bei 16 Bit dagegen nicht. Minus")
    print("  unendlich ist in jedem Zahlenformat exakt")

    print("\nUND DER FALL, DEN MINUS UNENDLICH FALSCH MACHT")
    nothing = np.zeros((1, 4), dtype=bool)
    with np.errstate(invalid="ignore"):
        blocked = masked_before(SCORES[:1], nothing)
    print("  eine Zeile, in der jede Position mit -inf maskiert ist")
    print("  ", blocked)
    print("  jeder Score ist -inf, also ist die Summe null, die Division")
    print("  undefiniert. Eine ganz aufgefüllte Zeile braucht einen")
    print("  Sonderfall, deshalb prüfen Bibliotheken genau darauf")


if __name__ == "__main__":
    main()
