"""Chapter 25. Encoder-only models: read everything, predict the gaps.

No causal mask, so every position sees the whole sentence in both
directions. That makes next-word prediction useless as a training task,
so a different one is needed.
"""
import numpy as np

from ch08_bpe import CORPUS
from ch09_embeddings import build_vocabulary

rng = np.random.default_rng(0)

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

SENTENCE = "the cat drank the milk".split()


def mask_tokens(tokens, rate=0.15, seed=0):
    """The training task: hide some tokens and ask for them back.

    Eight times in ten the token is replaced by a marker, once by a
    random word, and once left alone. The last two cases stop the model
    learning that the marker is the only place an answer is needed.
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
    """What each position may read. Everything, in both directions."""
    return np.ones((n, n), dtype=bool)


def causal_view(n):
    return np.tril(np.ones((n, n), dtype=bool))


def main():
    n = len(SENTENCE)
    print("WHAT EACH POSITION MAY READ")
    print("  encoder, bidirectional        decoder, causal")
    both, only = bidirectional_view(n), causal_view(n)
    for i in range(n):
        left = " ".join("y" if v else "." for v in both[i])
        right = " ".join("y" if v else "." for v in only[i])
        print(f"  {left}                   {right}")

    print("\nWHY NEXT-WORD PREDICTION CANNOT BE THE TASK HERE")
    print("  position 2 can see position 3, so asking it to predict")
    print("  position 3 is asking it to copy something it already has")

    print("\nMASKED LANGUAGE MODELLING, THE TASK THAT WORKS")
    for seed in [1, 4]:
        shown, targets = mask_tokens(SENTENCE, rate=0.4, seed=seed)
        wanted = sorted(targets.items())
        answers = ", ".join(f"{i}: {w}" for i, w in wanted)
        print(f"  input   {' '.join(shown)}")
        print(f"  wanted  {answers}")

    print("\nWHAT THE TWO ARCHITECTURES ARE FOR")
    print("  encoder-only    judging a whole text: is this review good,")
    print("                  do these two sentences agree, which span")
    print("                  of this document answers the question")
    print("  decoder-only    producing text one token at a time")

    print("\nA CLASSIFICATION HEAD")
    d_model, classes = 24, 3
    hidden = rng.standard_normal((n, d_model))
    head = rng.standard_normal((d_model, classes)) / np.sqrt(d_model)
    print(f"  hidden states {hidden.shape}")
    print("  take one position, usually a marker added at the front")
    print(f"  its vector {hidden[0].shape} times the head "
          f"{head.shape}")
    print(f"  gives {(hidden[0] @ head).shape}, one score per class")
    print(f"  scores {hidden[0] @ head}")
    print("  the rest of the positions are discarded for this task")


if __name__ == "__main__":
    main()
