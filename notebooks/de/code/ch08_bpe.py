"""Kapitel 8. Byte-Pair-Encoding, vor den Augen des Lesers trainiert.

Beginne mit einzelnen Zeichen. Finde immer wieder das häufigste
benachbarte Paar und verschmilz es zu einem neuen Symbol. Das ist der
ganze Algorithmus, und er erzeugt das Vokabular fast jedes Modells.
"""
from collections import Counter

CORPUS = (
    "die katze trank die milch und die katze schlief "
    "die maus trank die sahne und die maus schlief "
    "eine katze trank und eine maus schlief "
    "die milch und die sahne und die milch wieder"
)


def word_counts(text):
    """Jedes Wort wird ein Tupel aus Zeichen, mit Wortende-Marker."""
    counts = Counter(text.split())
    return {tuple(word) + ("_",): n for word, n in counts.items()}


def pair_counts(words):
    """Wie oft jedes benachbarte Paar im ganzen Korpus vorkommt."""
    pairs = Counter()
    for symbols, count in words.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += count
    return pairs


def merge(words, pair):
    """Jedes Vorkommen eines Paares durch ein Symbol ersetzen."""
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
    """Der Algorithmus, mit allen Verschmelzungen und Zwischenständen."""
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
    """Die Verschmelzungen in gelernter Reihenfolge anwenden."""
    symbols = tuple(word) + ("_",)
    for pair in learned:
        symbols = tuple(merge({symbols: 1}, pair))[0]
    return symbols


def vocabulary(learned, text):
    letters = {c for c in text if c != " "} | {"_"}
    return sorted(letters | {a + b for a, b in learned})


def main():
    print("DAS KORPUS, ALS ZEICHEN")
    words = word_counts(CORPUS)
    print(f"  {len(words)} verschiedene Wörter, {tokens_in(words)} "
          f"Tokens vor jeder Verschmelzung")

    learned, words, history = train(CORPUS, 12)

    print("\nZWÖLF VERSCHMELZUNGEN, HÄUFIGSTES PAAR ZUERST")
    print("  Nr.   neu        Anz.  Tokens übrig")
    for step, pair, count, total in history:
        joined = (pair[0] + pair[1]).replace("_", "␣")
        print(f"  {step:<5} {joined:<10} {count:<5} {total}")

    print("\nWAS DAS VOKABULAR JETZT ENTHÄLT")
    vocab = vocabulary(learned, CORPUS)
    print(f"  {len(vocab)} Symbole")
    line = "  "
    for symbol in (v.replace("_", "␣") for v in vocab):
        if len(line) + len(symbol) + 1 > 70:
            print(line)
            line = "  "
        line += " " + symbol
    print(line)

    print("\nWÖRTER MIT DIESEN VERSCHMELZUNGEN KODIEREN")
    for word in ["die", "katze", "maus", "trank", "elefant"]:
        pieces = encode(word, learned)
        shown = " ".join(p.replace("_", "␣") for p in pieces)
        count = f"{len(pieces)} Teil" + ("" if len(pieces) == 1 else "e")
        print(f"  {word:<10} {count + ':':<10} {shown}")

    print("\nMEHR VERSCHMELZUNGEN, WENIGER TEILE PRO WORT")
    print("  Verschm. Vokabular  Tokens für das Korpus")
    for n in [0, 5, 10, 20, 40, 80]:
        got, ended, _ = train(CORPUS, n)
        print(f"  {n:<8} {len(vocabulary(got, CORPUS)):<11} "
              f"{tokens_in(ended)}")


if __name__ == "__main__":
    main()
