"""Kapitel 33. Ein Bild ist eine Sequenz, sobald man es zerschneidet.

Nichts in der Architektur erwähnt Sprache. Die Attention arbeitet auf
einer Menge von Vektoren. Alles, was sich in eine Menge von Vektoren
verwandeln lässt, kann durch denselben Stapel, und diese Datei erledigt
das Verwandeln.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def patchify(image, patch):
    """Ein Bild in quadratische Patches schneiden, jedes flachdrücken."""
    height, width, channels = image.shape
    assert height % patch == 0 and width % patch == 0, "geht nicht auf"
    rows, cols = height // patch, width // patch
    out = image.reshape(rows, patch, cols, patch, channels)
    out = out.transpose(0, 2, 1, 3, 4)
    return out.reshape(rows * cols, patch * patch * channels)


def patch_embedding_parameters(patch, channels, d_model):
    return patch * patch * channels * d_model + d_model


def sequence_length(size, patch):
    return (size // patch) ** 2


def main():
    size, patch, channels, d_model = 224, 16, 3, 768
    image = np.random.default_rng(0).random((size, size, channels))

    print("AUS EINEM BILD WIRD EINE SEQUENZ")
    tokens = patchify(image, patch)
    print(f"  Bild         {image.shape}")
    print(f"  Patches      {tokens.shape}, eine Zeile pro Patch")
    print(f"  jede Zeile: {patch} x {patch} x {channels}, flach")
    print(f"  eine lineare Schicht: {tokens.shape[1]} auf {d_model}")
    print(f"  was der Stapel sieht: ({tokens.shape[0]}, {d_model})")
    print("  dieselbe Form, die ein Satz aus 196 Tokens ergibt")

    print("\nDIE GRÖSSE DER PATCHES BESTIMMT DIE SEQUENZLÄNGE")
    print("  Patch   Tokens   Kosten der Attention, relativ")
    base = sequence_length(size, 16) ** 2
    for p in [8, 14, 16, 32]:
        n = sequence_length(size, p)
        print(f"  {p:<7} {n:<8} {n * n / base:>8.2f}")
    print("  halbe Patches heißen viermal so viele Tokens und")
    print("  sechzehnmal so hohe Kosten für die Attention")

    print("\nDAS PATCH-EMBEDDING IST EINE KLEINE MATRIX")
    count = patch_embedding_parameters(patch, channels, d_model)
    print(f"  {patch}x{patch}x{channels} auf {d_model}: {count:_} "
          f"Parameter")
    print(f"  ein Transformer-Block dieser Breite: "
          f"{4 * d_model ** 2 + 8 * d_model ** 2:_}")
    print("  der Teil, der mit Bildern umgeht, ist der kleinste")

    print("\nWAS SICH ÄNDERT UND WAS NICHT")
    print("  anders   wie die Eingabe zerschnitten wird, die Positionen")
    print("  gleich   die Attention, das Feedforward-Netz, die Normen,")
    print("           die Residuen, der Block, der Stapel")

    print("\nANDERE ARTEN VON EINGABE, DIESELBE BEHANDLUNG")
    others = [("Audio", "25 ms eines Spektrogramms", 3000),
              ("Video", "Patch eines Einzelbilds", 9408),
              ("Protein", "Aminosäure", 350)]
    for kind, unit, count_ in others:
        print(f"  {kind:<8} ein Token pro {unit:<26} etwa {count_}")


if __name__ == "__main__":
    main()
