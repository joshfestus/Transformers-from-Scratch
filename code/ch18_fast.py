"""Chapter 18. What attention costs, and the four ways round it.

Every position against every position is a square number of
comparisons. This file measures that cost in memory and in time, then
measures what the usual remedies actually save.
"""
import time

import numpy as np

rng = np.random.default_rng(0)


def softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def score_memory(length, heads=32, bytes_each=2):
    """Bytes held by the score matrix, for one layer and one sequence."""
    return heads * length * length * bytes_each


def time_attention(length, width=64, repeats=3):
    """Wall clock for one head at this sequence length."""
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
    """Bytes of keys and values kept during generation."""
    kv_heads = heads if groups is None else groups
    return 2 * layers * kv_heads * length * d_head * bytes_each


def work_without_cache(steps):
    """Key and value rows recomputed if nothing is kept between steps."""
    return sum(range(1, steps + 1))


def work_with_cache(steps):
    """One new row per step, and the rest read back."""
    return steps


def main():
    print("THE SCORE MATRIX, ONE LAYER, 32 HEADS, HALF PRECISION")
    print("  length     scores held")
    for length in [512, 2048, 8192, 32768]:
        mb = score_memory(length) / 1e6
        shown = f"{mb:,.0f} MB" if mb < 1000 else f"{mb / 1000:,.1f} GB"
        print(f"  {length:<10} {shown}")

    print("\nMEASURED TIME FOR ONE HEAD, ON THIS MACHINE")
    print("  length   seconds     times longer   times the length")
    base = time_attention(256)
    for length in [256, 512, 1024, 2048]:
        taken = time_attention(length)
        print(f"  {length:<8} {taken:<11.4f} {taken / base:<14.1f} "
              f"{length / 256:.0f}")
    print("  doubling the length roughly quadruples the work")

    print("\nGENERATING 1000 TOKENS, KEY AND VALUE ROWS COMPUTED")
    without = work_without_cache(1000)
    with_cache = work_with_cache(1000)
    print(f"  without a cache  {without:,}")
    print(f"  with a cache     {with_cache:,}")
    print(f"  saved            {without / with_cache:.0f} times the work")

    print("\nWHAT THE CACHE COSTS, 32 LAYERS, 8192 TOKENS")
    print("  scheme              key and value heads   cache")
    for name, groups in [("multi-head", None), ("grouped, 8", 8),
                         ("multi-query", 1)]:
        size = cache_sizes(8192, groups=groups)
        heads = 32 if groups is None else groups
        print(f"  {name:<19} {heads:<21} {size / 1e9:.2f} GB")
    print("  the queries stay at 32 heads in all three")


if __name__ == "__main__":
    main()
