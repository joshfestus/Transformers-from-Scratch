"""Chapter 11. What a fixed-size summary can and cannot hold.

Before attention, a sequence model compressed everything it had read
into one vector of fixed width and handed that to whatever came next.
This file measures how much survives that compression, and what happens
instead when nothing is thrown away.
"""
import numpy as np

rng = np.random.default_rng(0)


def unit_vectors(count, width):
    """Random directions, each of length one."""
    v = rng.standard_normal((count, width))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def summary_test(length, width, trials=4000):
    """Add every vector into one, then ask what that summary still knows.

    The question is whether a vector was among those added. Score a probe
    by its dot product with the summary: something that was added scores
    about one plus interference from the rest, something that was not
    scores about nought plus the same interference. The test is whether
    the two can still be told apart.
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
    """The same question, keeping every vector instead of adding up."""
    hits = 0
    for _ in range(trials):
        items = unit_vectors(length + 1, width)
        kept = items[:length]
        stored = (kept @ items[rng.integers(length)]).max()
        novel = (kept @ items[length]).max()
        hits += int(stored > novel)
    return hits / trials


def noise_growth(length, width):
    """The size of the interference against the signal, in theory."""
    return np.sqrt((length - 1) / width)


def main():
    width = 32
    print(f"IS THIS VECTOR IN THERE? SUMMARY OF WIDTH {width}")
    print("  items   from the summary   keeping them all   noise")
    for length in [2, 4, 8, 16, 32, 64, 128]:
        summed = summary_test(length, width)
        kept = attention_test(length, width)
        print(f"  {length:<7} {summed:<18.3f} {kept:<18.3f} "
              f"{noise_growth(length, width):.2f}")

    print("\nA WIDER SUMMARY HOLDS MORE, IN PROPORTION")
    print("  width   items   answered correctly   noise")
    for w in [32, 128, 512, 2048]:
        print(f"  {w:<7} {64:<7} {summary_test(64, w):<20.3f} "
              f"{noise_growth(64, w):.2f}")

    print("\nWHAT A SEQUENCE COSTS, IN STEPS THAT CANNOT OVERLAP")
    print("  length   recurrent steps   attention steps")
    for length in [10, 100, 1000]:
        print(f"  {length:<8} {length:<17} {1}")


if __name__ == "__main__":
    main()
