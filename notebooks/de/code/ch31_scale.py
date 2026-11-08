"""Kapitel 31. Ein Modell genau zählen, und was sein Betrieb kostet.

Jede Zahl hier stammt aus einer Formel, angewendet auf veröffentlichte
Hyperparameter, nicht aus einer Tabelle, die jemand anderes erstellt hat.
"""
# Name, Schichten, Breite, Feedforward-Breite, Vokabular, Kontext, Bauart
# die Bauart sagt, welche der beiden Anordnungen aus diesem Buch das
# Modell benutzt: "plain" ist die aus Kapitel 23, "gated" die aus Kap. 19
MODELS = [
    ("klein", 12, 768, 3072, 50257, 1024, "plain"),
    ("mittel", 24, 1024, 4096, 50257, 1024, "plain"),
    ("groß", 36, 1280, 5120, 50257, 1024, "plain"),
    ("sieben Mrd.", 32, 4096, 11008, 32000, 4096, "gated"),
]


def block_parameters(d_model, d_ff, style="plain"):
    """Die Formel aus Kapitel 23, mit der gegateten Variante."""
    attention = 4 * d_model * d_model
    if style == "gated":
        feedforward = 3 * d_model * d_ff  # drei Matrizen, ohne Bias
        norms = 2 * d_model               # RMSNorm: nur ein Gain
    else:
        feedforward = 2 * d_model * d_ff + d_ff + d_model
        norms = 4 * d_model               # zwei Normen mit Gain und Bias
    return attention + feedforward + norms


def total_parameters(layers, d_model, d_ff, vocab, context,
                     style="plain"):
    total = layers * block_parameters(d_model, d_ff, style)
    total += vocab * d_model             # Embedding-Tabelle
    if style == "plain":
        total += context * d_model       # gelernte Positionen
        total += 2 * d_model             # letzte Norm, Gain, Bias
    else:
        total += vocab * d_model         # eigener Ausgabe-Head
        total += d_model                 # letzte RMSNorm
    return total


def memory_for_training(parameters, bytes_per_weight=2):
    """Gewichte, Gradienten und die beiden Mittelwerte von AdamW."""
    weights = parameters * bytes_per_weight
    gradients = parameters * bytes_per_weight
    adam = parameters * 4 * 2
    master = parameters * 4
    return {"Gewichte": weights, "Gradienten": gradients,
            "Optimierer": adam, "Masterkopie": master}


def flops_per_token(parameters, training=False):
    """Zwei Operationen pro Parameter vorwärts, dreimal so viel im
    Training."""
    return 6 * parameters if training else 2 * parameters


def main():
    print("PARAMETER, AUS DER FORMEL")
    print("  Modell           Schicht Breite  Parameter     veröffentl.")
    published = ["124 Mio.", "355 Mio.", "774 Mio.", "6.7 Mrd."]
    for (name, layers, d, ff, vocab, ctx, style), claim in zip(
            MODELS, published):
        total = total_parameters(layers, d, ff, vocab, ctx, style)
        print(f"  {name:<16} {layers:<7} {d:<7} {total / 1e6:>9.1f} Mio. "
              f"{claim:>10}")

    print("\nWO SIE SITZEN, WÄHREND DAS MODELL WÄCHST")
    print("  Modell           Blöcke    Embedding     Anteil Blöcke")
    for name, layers, d, ff, vocab, ctx, style in MODELS:
        blocks = layers * block_parameters(d, ff, style)
        embed = vocab * d
        share = 100 * blocks / (blocks + embed)
        print(f"  {name:<16} {blocks / 1e6:>5.0f} Mio. "
              f"{embed / 1e6:>7.0f} Mio. {share:>11.0f} %")
    print("  das Embedding ist der Großteil eines kleinen Modells und")
    print("  fast nichts eines großen, weil es mit der Breite wächst und")
    print("  die Blöcke mit dem Quadrat der Breite")

    print("\nWAS DAS TRAINING VERLANGT, FÜR EINEN SATZ PARAMETER")
    name, layers, d, ff, vocab, ctx, style = MODELS[-1]
    total = total_parameters(layers, d, ff, vocab, ctx, style)
    print(f"  das Modell {name}, {total / 1e9:.2f} Mrd. Parameter")
    parts = memory_for_training(total)
    for label, size in parts.items():
        print(f"  {label:<14} {size / 1e9:>6.1f} GB")
    print(f"  {'Summe':<14} {sum(parts.values()) / 1e9:>6.1f} GB, vor "
          f"einer einzigen Aktivierung")

    print("\nARITHMETIK PRO TOKEN")
    print("  Modell           Inferenz         Training")
    for name, layers, d, ff, vocab, ctx, style in MODELS:
        total = total_parameters(layers, d, ff, vocab, ctx, style)
        print(f"  {name:<16} {flops_per_token(total) / 1e9:>7.2f} GFLOP "
              f"{flops_per_token(total, True) / 1e9:>9.2f} GFLOP")

    print("\nWAS EIN TRAININGSLAUF AN ARITHMETIK KOSTET")
    total = total_parameters(32, 4096, 11008, 32000, 4096, "gated")
    for tokens in [1e9, 1e11, 1e12]:
        work = flops_per_token(total, True) * tokens
        print(f"  {tokens:>7.0e} Tokens   {work:.2e} Operationen")


if __name__ == "__main__":
    main()
