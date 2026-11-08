"""Kapitel 25. Reine Encoder-Modelle: alles lesen, Lücken vorhersagen.

Keine kausale Maske, also sieht jede Position den ganzen Satz in beide
Richtungen. Damit taugt die Vorhersage des nächsten Wortes nicht als
Trainingsaufgabe, und es braucht eine andere.
"""
import numpy as np

from ch08_bpe import CORPUS
from ch09_embeddings import build_vocabulary

rng = np.random.default_rng(0)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

SENTENCE = "die katze trank die milch".split()


def mask_tokens(tokens, rate=0.15, seed=0):
    """Die Trainingsaufgabe: einige Tokens verstecken, zurückverlangen.

    In acht von zehn Fällen wird das Token durch einen Marker ersetzt,
    einmal durch ein zufälliges Wort und einmal so gelassen. Die letzten
    beiden Fälle verhindern, dass das Modell lernt, nur beim Marker werde
    eine Antwort gebraucht.
    """
    local = np.random.default_rng(seed)
    vocab = build_vocabulary(CORPUS)
    out, targets = list(tokens), {}
    for i, token in enumerate(tokens):
        if local.random() >= rate:
            continue
        targets[i] = token
        roll = local.random()
        if roll < 0.8:
            out[i] = "[MASK]"
        elif roll < 0.9:
            out[i] = str(local.choice(vocab))
    return out, targets


def bidirectional_view(n):
    """Was jede Position lesen darf. Alles, in beide Richtungen."""
    return np.ones((n, n), dtype=bool)


def causal_view(n):
    return np.tril(np.ones((n, n), dtype=bool))


def main():
    n = len(SENTENCE)
    print("WAS JEDE POSITION LESEN DARF")
    print("  Encoder, bidirektional        Decoder, kausal")
    both, only = bidirectional_view(n), causal_view(n)
    for i in range(n):
        left = " ".join("j" if v else "." for v in both[i])
        right = " ".join("j" if v else "." for v in only[i])
        print(f"  {left}                   {right}")

    print("\nWARUM DAS NÄCHSTE WORT HIER KEINE AUFGABE SEIN KANN")
    print("  Position 2 sieht Position 3, sie Position 3 vorhersagen zu")
    print("  lassen hieße, etwas abschreiben lassen, das sie schon hat")

    print("\nMASKED LANGUAGE MODELLING, DIE AUFGABE, DIE FUNKTIONIERT")
    for seed in [1, 4]:
        shown, targets = mask_tokens(SENTENCE, rate=0.4, seed=seed)
        wanted = sorted(targets.items())
        answers = ", ".join(f"{i}: {w}" for i, w in wanted)
        print(f"  Eingabe  {' '.join(shown)}")
        print(f"  gesucht  {answers}")

    print("\nWOFÜR DIE BEIDEN ARCHITEKTUREN DA SIND")
    print("  nur Encoder     einen ganzen Text beurteilen: Ist diese")
    print("                  Rezension gut, stimmen diese zwei Sätze")
    print("                  überein, welche Stelle beantwortet die")
    print("                  Frage")
    print("  nur Decoder     Text erzeugen, ein Token nach dem anderen")

    print("\nEIN KLASSIFIKATIONS-HEAD")
    d_model, classes = 24, 3
    hidden = rng.standard_normal((n, d_model))
    head = rng.standard_normal((d_model, classes)) / np.sqrt(d_model)
    print(f"  verdeckte Zustände {hidden.shape}")
    print("  nimm eine Position, meist einen Marker, vorne angefügt")
    print(f"  ihr Vektor {hidden[0].shape} mal der Head "
          f"{head.shape}")
    print(f"  ergibt {(hidden[0] @ head).shape}, ein Score pro Klasse")
    print(f"  Scores {hidden[0] @ head}")
    print("  die übrigen Positionen werden für diese Aufgabe verworfen")


if __name__ == "__main__":
    main()
