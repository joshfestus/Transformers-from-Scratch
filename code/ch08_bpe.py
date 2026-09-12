"""Chapter 8. Byte pair encoding, trained in front of the reader.

Start from single characters. Repeatedly find the most common adjacent
pair and merge it into one new symbol. That is the whole algorithm, and
it is what produces the vocabulary of nearly every model in use.
"""
from collections import Counter

CORPUS = (
    "the cat drank the milk and the cat slept "
    "the dog drank the water and the dog slept "
    "a cat and a dog drank and slept "
    "the milk and the water and the milk again"
)


def word_counts(text):
    """Each word becomes a tuple of characters, with a word-end marker."""
    counts = Counter(text.split())
    return {tuple(word) + ("_",): n for word, n in counts.items()}


def pair_counts(words):
    """How often each adjacent pair occurs, across the whole corpus."""
    pairs = Counter()
    for symbols, count in words.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += count
    return pairs


def merge(words, pair):
    """Replace every occurrence of one pair with a joined symbol."""
    joined = pair[0] + pair[1]
    out = {}
    for symbols, count in words.items():
        new, i = [], 0
        while i < len(symbols):
            if (i < len(symbols) - 1
                    and (symbols[i], symbols[i + 1]) == pair):
                new.append(joined)
                i += 2
            else:
                new.append(symbols[i])
                i += 1
        out[tuple(new)] = count
    return out


def train(text, merges):
    """Run the algorithm, keeping the merges and each step's state."""
    words = word_counts(text)
    history, learned = [], []
    for step in range(merges):
        pairs = pair_counts(words)
        if not pairs:
            break
        pair, count = pairs.most_common(1)[0]
        words = merge(words, pair)
        learned.append(pair)
        history.append((step + 1, pair, count, tokens_in(words)))
    return learned, words, history


def tokens_in(words):
    return sum(len(symbols) * count for symbols, count in words.items())


def encode(word, learned):
    """Apply the merges to a new word, in the order they were learned."""
    symbols = tuple(word) + ("_",)
    for pair in learned:
        symbols = tuple(merge({symbols: 1}, pair))[0]
    return symbols


def vocabulary(learned, text):
    letters = {c for c in text if c != " "} | {"_"}
    return sorted(letters | {a + b for a, b in learned})


def main():
    print("THE CORPUS, AS CHARACTERS")
    words = word_counts(CORPUS)
    print(f"  {len(words)} distinct words, {tokens_in(words)} tokens "
          f"before any merging")

    learned, words, history = train(CORPUS, 12)

    print("\nTWELVE MERGES, MOST COMMON PAIR FIRST")
    print("  step  merged     seen  tokens left")
    for step, pair, count, total in history:
        joined = (pair[0] + pair[1]).replace("_", "␣")
        print(f"  {step:<5} {joined:<10} {count:<5} {total}")

    print("\nWHAT THE VOCABULARY HOLDS NOW")
    vocab = vocabulary(learned, CORPUS)
    print(f"  {len(vocab)} symbols")
    line = "  "
    for symbol in (v.replace("_", "␣") for v in vocab):
        if len(line) + len(symbol) + 1 > 70:
            print(line)
            line = "  "
        line += " " + symbol
    print(line)

    print("\nENCODING WORDS WITH THOSE MERGES")
    for word in ["the", "cat", "dog", "drank", "elephant"]:
        pieces = encode(word, learned)
        shown = " ".join(p.replace("_", "␣") for p in pieces)
        print(f"  {word:<10} {len(pieces)} pieces:  {shown}")

    print("\nMORE MERGES MEANS FEWER PIECES PER WORD")
    print("  merges  vocabulary  tokens for the corpus")
    for n in [0, 5, 10, 20, 40, 80]:
        got, ended, _ = train(CORPUS, n)
        print(f"  {n:<7} {len(vocabulary(got, CORPUS)):<11} "
              f"{tokens_in(ended)}")


if __name__ == "__main__":
    main()
