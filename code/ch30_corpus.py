"""Chapter 30. A corpus whose rules are known, so learning can be checked.

Training on found text makes it hard to say whether a model learned
anything, because nobody can state what the right answer was. This
corpus is generated from a small grammar written out below, so the
structure the model has to discover is known exactly, and generated
text can be checked against it rather than admired.
"""
import random

DETERMINERS = ["the", "a", "every"]
ADJECTIVES = ["small", "grey", "hungry", "quiet", "old"]
NOUNS = ["cat", "dog", "bird", "fox", "child", "farmer"]
VERBS = ["watched", "chased", "found", "carried", "followed"]
PLACES = ["garden", "river", "field", "kitchen", "hill"]


def agree(determiner, following):
    """`a` becomes `an` before a vowel. A rule the model has to learn."""
    if determiner == "a" and following[0] in "aeiou":
        return "an"
    return determiner


def noun_phrase(rng):
    """determiner, an optional adjective, then a noun."""
    words = []
    if rng.random() < 0.5:
        words.append(rng.choice(ADJECTIVES))
    words.append(rng.choice(NOUNS))
    return [agree(rng.choice(DETERMINERS), words[0])] + words


def sentence(rng):
    """noun phrase, verb, noun phrase, and sometimes a place."""
    parts = noun_phrase(rng) + [rng.choice(VERBS)] + noun_phrase(rng)
    if rng.random() < 0.4:
        place = rng.choice(PLACES)
        parts += ["in", agree(rng.choice(DETERMINERS), place), place]
    return " ".join(parts) + "."


def build(count=1200, seed=0):
    rng = random.Random(seed)
    return "\n".join(sentence(rng) for _ in range(count)) + "\n"


def is_grammatical(line):
    """Check one line against the grammar that produced the data."""
    line = line.strip()
    if not line.endswith("."):
        return False
    words = line[:-1].split()
    i = 0

    def phrase():
        nonlocal i
        if i >= len(words) or words[i] not in DETERMINERS + ["an"]:
            return False
        determiner = words[i]
        i += 1
        if i >= len(words):
            return False
        mismatch = agree("a", words[i]) != determiner
        if determiner in ("a", "an") and mismatch:
            return False
        if words[i] in ADJECTIVES:
            i += 1
        if i >= len(words) or words[i] not in NOUNS:
            return False
        i += 1
        return True

    if not phrase():
        return False
    if i >= len(words) or words[i] not in VERBS:
        return False
    i += 1
    if not phrase():
        return False
    if i < len(words):
        if words[i] != "in":
            return False
        i += 1
        if i >= len(words) or words[i] not in DETERMINERS + ["an"]:
            return False
        determiner = words[i]
        i += 1
        if i >= len(words) or words[i] not in PLACES:
            return False
        mismatch = agree("a", words[i]) != determiner
        if determiner in ("a", "an") and mismatch:
            return False
        i += 1
    return i == len(words)


def main():
    text = build()
    lines = text.strip().split("\n")
    print("A CORPUS GENERATED FROM A GRAMMAR")
    print(f"  {len(lines)} sentences, {len(text):,} characters")
    print(f"  {len(set(text))} distinct characters")
    print("  first four sentences")
    for line in lines[:4]:
        print(f"    {line}")

    print("\nEVERY SENTENCE IN IT PASSES THE CHECKER, BY CONSTRUCTION")
    passed = sum(is_grammatical(line) for line in lines)
    print(f"  {passed} of {len(lines)}")

    print("\nAND THE CHECKER REJECTS WHAT IT SHOULD")
    for bad in ["cat the watched dog.", "the cat watched.",
                "the cat watched the dog in.", "the small grey cat ran.",
                "a old farmer chased a bird."]:
        print(f"  {bad:<32} {is_grammatical(bad)}")


if __name__ == "__main__":
    main()
