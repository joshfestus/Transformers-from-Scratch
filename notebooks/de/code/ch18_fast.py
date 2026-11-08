"""Kapitel 18. Was die Attention kostet, und vier Wege drumherum.

Jede Position gegen jede Position ist eine quadratische Zahl von
Vergleichen. Diese Datei misst diese Kosten an Speicher und Zeit und
dann, was die üblichen Abhilfen tatsächlich einsparen.
"""
import time

import numpy as np

rng = np.random.default_rng(0)


def softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def score_memory(length, heads=32, bytes_each=2):
    """Bytes der Score-Matrix, für eine Schicht und eine Sequenz."""
    return heads * length * length * bytes_each


def time_attention(length, width=64, repeats=3):
    """Echte Laufzeit für einen Head bei dieser Sequenzlänge."""
    q = rng.standard_normal((length, width))
    k = rng.standard_normal((length, width))
    v = rng.standard_normal((length, width))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        softmax(q @ k.T / np.sqrt(width)) @ v
        best = min(best, time.perf_counter() - start)
    return best


def cache_sizes(length, layers=32, heads=32, d_head=128, groups=None,
                bytes_each=2):
    """Bytes der Keys und Values, die beim Generieren bleiben."""
    kv_heads = heads if groups is None else groups
    return 2 * layers * kv_heads * length * d_head * bytes_each


def work_without_cache(steps):
    """Key- und Value-Zeilen, neu berechnet, wenn nichts bleibt."""
    return sum(range(1, steps + 1))


def work_with_cache(steps):
    """Eine neue Zeile pro Schritt, der Rest wird gelesen."""
    return steps


def main():
    print("DIE SCORE-MATRIX, EINE SCHICHT, 32 HEADS, 16 BIT")
    print("  Länge      Speicher für die Scores")
    for length in [512, 2048, 8192, 32768]:
        mb = score_memory(length) / 1e6
        shown = f"{mb:.0f} MB" if mb < 1000 else f"{mb / 1000:.1f} GB"
        print(f"  {length:<10} {shown}")

    print("\nGEMESSENE ZEIT FÜR EINEN HEAD, AUF DIESEM RECHNER")
    print("  Länge    Sekunden    mal länger     mal die Länge")
    base = time_attention(256)
    for length in [256, 512, 1024, 2048]:
        taken = time_attention(length)
        print(f"  {length:<8} {taken:<11.4f} {taken / base:<14.1f} "
              f"{length / 256:.0f}")
    print("  doppelte Länge heißt ungefähr vierfache Arbeit")

    print("\n1000 TOKENS ERZEUGEN, BERECHNETE KEY- UND VALUE-ZEILEN")
    without = work_without_cache(1000)
    with_cache = work_with_cache(1000)
    print(f"  ohne Cache       {without:_}")
    print(f"  mit Cache        {with_cache:_}")
    print(f"  gespart          {without / with_cache:.0f}-mal die Arbeit")

    print("\nWAS DER CACHE KOSTET, 32 SCHICHTEN, 8192 TOKENS")
    print("  Verfahren           Key- und Value-Heads  Cache")
    for name, groups in [("Multi-Head", None), ("gruppiert, 8", 8),
                         ("Multi-Query", 1)]:
        size = cache_sizes(8192, groups=groups)
        heads = 32 if groups is None else groups
        print(f"  {name:<19} {heads:<21} {size / 1e9:.2f} GB")
    print("  die Queries bleiben in allen dreien bei 32 Heads")


if __name__ == "__main__":
    main()
