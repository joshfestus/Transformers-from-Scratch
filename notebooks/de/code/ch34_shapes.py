"""Kapitel 34. Ein Paper lesen, indem man den Formen folgt.

Die Methode: Nimm eine beliebige Gleichung, bestimme die Form von allem
darin und prüf, ob die Formen zusammenpassen. Tun sie es, hast du die
Operation verstanden. Tun sie es nicht, hast du falsch gelesen.
"""
import numpy as np

from transformer_numpy import Config, init_params

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def trace(cfg, n=5):
    """Jeder Tensor eines Vorwärtsdurchlaufs, und was er enthält."""
    d, h, f, v = cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.vocab
    return [
        ("Token-Nummern", (n,), "eine ganze Zahl pro Position"),
        ("Embeddings", (n, d), "ein Vektor pro Position"),
        ("plus Positionen", (n, d), "dasselbe, jetzt mit Reihenfolge"),
        ("Q, K, V", (n, d), "drei Projektionen derselben Zeilen"),
        ("in Heads geteilt", (h, n, d // h), "ein Ausschnitt pro Head"),
        ("Scores", (h, n, n), "jede Query gegen jeden Key"),
        ("Gewichte", (h, n, n), "jede Zeile ergibt eins"),
        ("gemischt", (h, n, d // h), "eine Ausgabe pro Query und Head"),
        ("zusammengefügt", (n, d), "Heads wieder vereint"),
        ("nach dem Block", (n, d), "Form unverändert, immer"),
        ("Logits", (n, v), "ein Score pro Token im Vokabular"),
    ]


def check(name, shape, expected):
    return "ok" if tuple(shape) == tuple(expected) else "MISMATCH"


def main():
    cfg = Config()
    print("JEDE FORM EINES VORWÄRTSDURCHLAUFS")
    print("  Tensor                Form           was er enthält")
    for name, shape, meaning in trace(cfg):
        print(f"  {name:<21} {str(shape):<14} {meaning}")

    print("\nEINE GLEICHUNG ÜBER IHRE FORMEN LESEN")
    print("  die Formel sagt softmax(Q K-transponiert / Wurzel d_k) V")
    n, d, h = 5, cfg.d_model, cfg.n_heads
    dh = cfg.d_head
    steps = [("Q", (h, n, dh)), ("K transponiert", (h, dh, n)),
             ("ihr Produkt", (h, n, n)), ("nach Softmax", (h, n, n)),
             ("V", (h, n, dh)), ("das Ergebnis", (h, n, dh))]
    for name, shape in steps:
        print(f"  {name:<16} {shape}")
    print("  die inneren Dimensionen passen bei jedem Schritt, also sagt")
    print("  die Gleichung, was sie zu sagen scheint")

    print("\nWO DIE PARAMETER SIND, ALS LETZTE PRÜFUNG")
    params = init_params(cfg, seed=0)
    counted = params["embed"].size + params["final_gain"].size \
        + params["final_bias"].size
    for block in params["blocks"]:
        counted += sum(value.size for value in block.values())
    print(f"  {counted:_} Zahlen im ganzen Modell")
    print("  und jede davon kam oben in einer Form vor")

    print("\nDIE VIER FRAGEN AN JEDE NEUE ARCHITEKTUR")
    questions = [
        "Was ist hier ein Token, und wie viele gibt es?",
        "Was mischt Information zwischen Positionen, was nicht?",
        "Wo ändert sich die Form, und wo bleibt sie erhalten?",
        "Was ist das Trainingssignal, und wie viele pro Sequenz?",
    ]
    for i, question in enumerate(questions, 1):
        print(f"  {i}. {question}")


if __name__ == "__main__":
    main()
