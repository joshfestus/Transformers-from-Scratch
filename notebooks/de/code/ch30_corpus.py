"""Kapitel 30. Ein Korpus mit bekannten Regeln, um Lernen zu prüfen.

Wer auf gefundenem Text trainiert, kann schwer sagen, ob ein Modell
etwas gelernt hat, denn niemand kann sagen, was die richtige Antwort
gewesen wäre. Dieses Korpus wird aus einer kleinen Grammatik erzeugt,
die unten ausgeschrieben steht. Die Struktur, die das Modell entdecken
muss, ist also genau bekannt, und erzeugter Text lässt sich an ihr
prüfen, statt ihn nur zu bewundern.
"""
import random

NOUNS = ["hund", "fuchs", "vogel", "bauer", "katze", "ziege", "kind",
         "pferd"]
ADJECTIVES = ["klein", "grau", "hungrig", "still", "alt"]
VERBS = ["sah", "jagte", "fand", "trug", "weckte"]
PLACES = ["garten", "wald", "feld", "scheune", "stall"]
GENDER = {"hund": "m", "fuchs": "m", "vogel": "m", "bauer": "m",
          "katze": "f", "ziege": "f", "kind": "n", "pferd": "n",
          "garten": "m", "wald": "m", "feld": "n", "scheune": "f",
          "stall": "m",
          # die Orte der verschobenen Aufgabe aus Kapitel 32
          "hafen": "m", "markt": "m", "bahnhof": "m", "tal": "n",
          "schule": "f"}

# der Artikel je Bestimmungswort, Fall und Geschlecht
ARTICLES = {
    ("der", "nom"): {"m": "der", "f": "die", "n": "das"},
    ("der", "akk"): {"m": "den", "f": "die", "n": "das"},
    ("ein", "nom"): {"m": "ein", "f": "eine", "n": "ein"},
    ("ein", "akk"): {"m": "einen", "f": "eine", "n": "ein"},
    ("jeder", "nom"): {"m": "jeder", "f": "jede", "n": "jedes"},
    ("jeder", "akk"): {"m": "jeden", "f": "jede", "n": "jedes"},
}
# die Endung des Adjektivs nach jedem Bestimmungswort
ENDINGS = {
    ("der", "nom"): {"m": "e", "f": "e", "n": "e"},
    ("der", "akk"): {"m": "en", "f": "e", "n": "e"},
    ("ein", "nom"): {"m": "er", "f": "e", "n": "es"},
    ("ein", "akk"): {"m": "en", "f": "e", "n": "es"},
    ("jeder", "nom"): {"m": "e", "f": "e", "n": "e"},
    ("jeder", "akk"): {"m": "en", "f": "e", "n": "e"},
}
KINDS = ["der", "ein", "jeder"]


def noun_phrase(rng, case):
    """Artikel, ein mögliches Adjektiv, dann ein Substantiv.

    Artikel und Endung hängen vom Geschlecht des Substantivs ab, das
    erst danach kommt, und vom Fall. Diese Regel muss das Modell lernen.
    """
    noun = rng.choice(NOUNS)
    kind, gender = rng.choice(KINDS), GENDER[noun]
    words = [ARTICLES[(kind, case)][gender]]
    if rng.random() < 0.5:
        ending = ENDINGS[(kind, case)][gender]
        words.append(rng.choice(ADJECTIVES) + ending)
    return words + [noun]


def place_phrase(rng):
    """Ein Ort: im garten, aber in der scheune."""
    place = rng.choice(PLACES)
    if GENDER[place] == "f":
        return ["in", "der", place]
    return ["im", place]


def sentence(rng):
    """Subjekt, Verb, Objekt, und manchmal ein Ort."""
    parts = (noun_phrase(rng, "nom") + [rng.choice(VERBS)]
             + noun_phrase(rng, "akk"))
    if rng.random() < 0.4:
        parts += place_phrase(rng)
    return " ".join(parts) + "."


def build(count=1200, seed=0):
    rng = random.Random(seed)
    return "\n".join(sentence(rng) for _ in range(count)) + "\n"


def agrees(article, adjective, noun, case):
    """Passen Artikel und Endung zu Geschlecht und Fall?"""
    gender = GENDER[noun]
    for kind in KINDS:
        if ARTICLES[(kind, case)][gender] != article:
            continue
        if adjective is None:
            return True
        ending = ENDINGS[(kind, case)][gender]
        if adjective in (a + ending for a in ADJECTIVES):
            return True
    return False


def is_grammatical(line):
    """Eine Zeile an der Grammatik prüfen, die die Daten erzeugt hat."""
    line = line.strip()
    if not line.endswith("."):
        return False
    words = line[:-1].split()
    i = 0

    def phrase(case):
        nonlocal i
        if i >= len(words):
            return False
        article = words[i]
        i += 1
        adjective = None
        if i < len(words) and words[i] not in NOUNS:
            adjective = words[i]
            i += 1
        if i >= len(words) or words[i] not in NOUNS:
            return False
        noun = words[i]
        i += 1
        return agrees(article, adjective, noun, case)

    if not phrase("nom"):
        return False
    if i >= len(words) or words[i] not in VERBS:
        return False
    i += 1
    if not phrase("akk"):
        return False
    if i < len(words):
        if words[i:i + 2] == ["in", "der"]:
            i += 2
            if i >= len(words) or words[i] not in PLACES:
                return False
            if GENDER[words[i]] != "f":
                return False
        elif words[i] == "im":
            i += 1
            if i >= len(words) or words[i] not in PLACES:
                return False
            if GENDER[words[i]] == "f":
                return False
        else:
            return False
        i += 1
    return i == len(words)


def main():
    text = build()
    lines = text.strip().split("\n")
    print("EIN KORPUS, AUS EINER GRAMMATIK ERZEUGT")
    print(f"  {len(lines)} Sätze, {len(text):_} Zeichen")
    print(f"  {len(set(text))} verschiedene Zeichen")
    print("  die ersten vier Sätze")
    for line in lines[:4]:
        print(f"    {line}")

    print("\nJEDER SATZ DARIN BESTEHT DIE PRÜFUNG, SCHON VON BAUART")
    passed = sum(is_grammatical(line) for line in lines)
    print(f"  {passed} von {len(lines)}")

    print("\nUND DIE PRÜFUNG LEHNT AB, WAS SIE ABLEHNEN SOLL")
    for bad in ["katze die sah den hund.", "die katze sah.",
                "die katze sah den hund im.",
                "die kleine graue katze lief.",
                "der katze sah den hund.", "die katze sah der hund.",
                "ein kleine hund sah die katze."]:
        print(f"  {bad:<32} {is_grammatical(bad)}")


if __name__ == "__main__":
    main()
