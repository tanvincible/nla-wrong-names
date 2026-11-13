"""Broad name detection: does an explanation name ANY entity of a type, not just one of our 10?

  country     every country in GeoNames plus common short names (case-sensitive)
  city        GeoNames cities over 300k people, minus names that are ordinary English words (case-sensitive)
  person      spaCy NER PERSON spans (en_core_web_sm)
  animal, food, instrument   WordNet hyponyms, keeping a word only where that sense is its commonest use

classify(text, passage_text, entity) -> (narrow, broad), each "true" / "other" / "neither":
  narrow  other = names one of the 9 other listed entities of the type
  broad   other = names any same-type entity that is not the true one and does not occur in the passage
          (a same-type name that IS in the passage is real content, not a wrong name, so it is ignored)
Known limits: the small NER model misses and invents some people; ambiguous words are dropped by the
frequency rule (so "bear" and "bass" are not detected); demonyms ("French") never count as a country.

NAMES_MODE=clean (the default) also drops noise found by hand-checking 30 flagged explanations: category
words (bird, mammal, food product), meals (dinner, breakfast), words for people (humans), and "person"
matches that are not proper nouns (ed, song). NAMES_MODE=raw gives the first, noisier detector.
"""
import functools
import os
import re

CLEAN = os.environ.get("NAMES_MODE", "clean") == "clean"
# the macOS word list, kept in the repo so name detection is identical on a Linux GPU box
WORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "words.txt")

from entities import ENTITIES, PROPER, TYPE, forms, named

GENERIC = {  # category words, not names of a specific thing
    "animal": "animal creature beast pet male female young adult predator prey game stock livestock mate giant "
              "feeder racer stray survivor pest critter fauna brute varmint darter".split(),
    "food": "food dish meal course fare snack treat feed produce drink beverage nutrient nutriment diet ration meat "
            "fruit vegetable seafood dessert sweet staple ingredient refreshment delicacy leftovers serving helping "
            "portion plate side special mess mix stuff liquid water pop spread cut joint side slice piece".split(),
    "instrument": "instrument keyboard string strings wind brass percussion woodwind section key keys pipe bones".split(),
}
if CLEAN:
    GENERIC["animal"] += "human humans man men woman women homo people person child children poultry wildlife cattle".split()
    GENERIC["food"] += ("breakfast lunch dinner supper brunch feast banquet picnic buffet foodstuff provisions groceries "
                        "sustenance cooking spirit spirits grub candy").split() + ["old fashioned", "food product", "breakfast food", "junk food", "fast food"]
BIG = 250  # clean mode: a WordNet entry with more kinds under it than this is a category, not a specific thing
ROOTS = {"animal": ["animal.n.01"], "food": ["food.n.01", "food.n.02"], "instrument": ["musical_instrument.n.01"]}
EXTRA_COUNTRIES = ["USA", "America", "Britain", "England", "Scotland", "Wales", "UK", "Holland", "Burma", "Persia",
                   "Czechoslovakia", "Yugoslavia", "USSR", "Soviet Union", "Korea", "Russia", "Iran", "Syria", "Vietnam"]


def _wordnet(typ):
    from nltk.corpus import wordnet as wn
    tree = set()
    for r in ROOTS[typ]:
        s = wn.synset(r)
        tree |= {s} | set(s.closure(lambda x: x.hyponyms()))
    out = set()
    if CLEAN:
        meal = wn.synset("meal.n.01")
        tree -= {meal} | set(meal.closure(lambda x: x.hyponyms()))
        tree = {s for s in tree if sum(1 for _ in s.closure(lambda x: x.hyponyms())) <= BIG}
    for syn in tree:
        for lem in syn.lemmas():
            name = lem.name()
            w = name.replace("_", " ").lower()
            if len(w) < 3 or w in GENERIC[typ] or not re.fullmatch(r"[a-z][a-z \-']+", w):
                continue
            if "_" not in name:  # single words: keep only if this type is the word's commonest use
                senses = wn.synsets(name)
                count = lambda inside: sum(l.count() for s in senses if (s in tree) == inside
                                           for l in s.lemmas() if l.name().lower() == name.lower())
                if count(False) > count(True) or (count(True) == 0 and senses[0] not in tree):
                    continue
            out.add(w)
    return out


@functools.lru_cache(None)
def _matcher(typ):
    """-> function text -> [(start, end, name)] for every name of this type in the text."""
    if typ == "person":
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        words = {w.strip() for w in open(WORDS)}
        particle = {"de", "van", "von", "der", "den", "la", "le", "da", "di", "del", "bin", "al", "el", "du"}

        def people(text):
            out = []
            for e in nlp(text).ents:
                m = re.match(r"[\s\"'(]*(.*?)(?:['’]s?)?[\s.,;:\"')]*$", e.text, re.S)
                span = m.group(1)
                toks = span.split()
                if e.label_ != "PERSON" or not toks:
                    continue
                # clean mode: a proper noun only, and a lone word must not be an ordinary English word
                if CLEAN and (not all(re.fullmatch(r"[A-Z][A-Za-z.'\-]+", t) or t in particle for t in toks)
                              or toks[0] in particle or toks[-1] in particle
                              or (len(toks) == 1 and (toks[0].lower() in words or len(toks[0]) < 3))):
                    continue
                start = e.start_char + m.start(1)
                out.append((start, start + len(span), span))
            return out
        return people
    if typ in ("country", "city"):
        import geonamescache
        gc = geonamescache.GeonamesCache()
        countries = {c["name"] for c in gc.get_countries().values()} | set(EXTRA_COUNTRIES)
        if typ == "country":
            names = countries
        else:
            words = {w.strip() for w in open(WORDS)}  # lowercase entries = ordinary words
            names = {c["name"] for c in gc.get_cities().values()
                     if c["population"] >= 300_000 and c["name"].lower() not in words and c["name"] not in countries}
        names = {n for n in names if re.fullmatch(r"[A-Z][\w .'\-]+", n)} | set(ENTITIES[typ])
        pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")\b")
        return lambda text: [(m.start(), m.end(), m.group(1)) for m in pat.finditer(text)]
    names = _wordnet(typ) | {f.lower() for e in ENTITIES[typ] for f in forms(e)}
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")(?:e?s)?\b", re.I)
    return lambda text: [(m.start(), m.end(), m.group(1).lower()) for m in pat.finditer(text)]


def broad_named(text, typ):
    return {n for _, _, n in _matcher(typ)(text)}


def _canon(typ):
    return {(f if typ in PROPER else f.lower()): e for e in ENTITIES[typ] for f in forms(e)}


def all_names(text, typ):
    """Canonical same-type names in a text, with no reference to any passage (for agreement between samples)."""
    canon, out = _canon(typ), set()
    for n in broad_named(text, typ) | named(text, typ):
        parts = [n] + (n.split() if typ == "person" else [])
        hit = [canon[p] for p in parts if p in canon]
        out.add(hit[0] if hit else n.split()[-1] if typ == "person" else n)
    return out


def name_spans(text, passage_text, entity, with_name=False):
    """[(start, end, "true" | "wrong")] for every same-type name occurrence; names the passage contains are skipped.
    with_name=True adds the canonical name (the one claims() reports) as a fourth element."""
    from entities import _PAT
    typ = TYPE[entity]
    low = passage_text if typ in PROPER else passage_text.lower()
    mine = {f if typ in PROPER else f.lower() for f in forms(entity)}
    canon = _canon(typ)
    spans = [(m.start(), m.end(), "true", entity) for m in _PAT[entity].finditer(text)]
    found = list(_matcher(typ)(text)) + [(m.start(), m.end(), e) for e in ENTITIES[typ] if e != entity
                                         for m in _PAT[e].finditer(text)]
    for a, b, n in found:
        parts = [n] + (n.split() if typ == "person" else [])
        if any(p in mine for p in parts) or any(re.search(r"\b" + re.escape(p), low) for p in parts):
            continue
        hit = [canon[p] for p in parts if p in canon]
        spans.append((a, b, "wrong", hit[0] if hit else n.split()[-1] if typ == "person" else n))
    out = []  # drop overlaps, longest first
    for sp in sorted(spans, key=lambda x: (x[0] - x[1], x[0])):
        if all(sp[1] <= o[0] or sp[0] >= o[1] for o in out):
            out.append(sp)
    return sorted(out) if with_name else sorted(sp[:3] for sp in out)


GENERIC_NOUN = {"country": ("country", "countries"), "city": ("city", "cities"), "person": ("person", "people"),
                "animal": ("animal", "animals"), "food": ("food", "foods"), "instrument": ("instrument", "instruments")}


def replace_names(text, passage_text, entity, kind):
    """Replace every `kind` ("true" or "wrong") same-type name with "another <type>", absorbing a leading article."""
    typ = TYPE[entity]
    for a, b, k in reversed(name_spans(text, passage_text, entity)):
        if k != kind:
            continue
        surf = text[a:b]
        plural = typ not in PROPER and surf.lower().endswith("s") and not surf.lower().endswith("ss") and len(surf) > 3
        rep = "other " + GENERIC_NOUN[typ][1] if plural else "another " + GENERIC_NOUN[typ][0]
        art = re.search(r"\b(a|an|the)\s+$", text[:a], re.I)
        if art:
            a = art.start()
        if a == 0 or text[:a].rstrip().endswith((".", "!", "?")) or (art and art.group(1)[0].isupper()):
            rep = rep.capitalize()
        text = text[:a] + rep + text[b:]
    return text


def claims(text, passage_text, entity):
    """-> (does the text name the true entity, set of wrong same-type names it gives).
    A wrong name is a same-type entity that is not the true one and does not occur in the passage. Names are
    canonical: one of our listed entities where it is one, a person's last word, otherwise the matched word."""
    typ = TYPE[entity]
    listed = named(text, typ)
    low = passage_text if typ in PROPER else passage_text.lower()
    mine = {f if typ in PROPER else f.lower() for f in forms(entity)}
    canon = _canon(typ)
    wrong = set()
    for n in broad_named(text, typ) | (listed - {entity}):
        parts = [n] + (n.split() if typ == "person" else [])
        if any(p in mine for p in parts) or any(re.search(r"\b" + re.escape(p), low) for p in parts):
            continue  # the true entity, or something the passage really says
        hit = [canon[p] for p in parts if p in canon]
        wrong.add(hit[0] if hit else n.split()[-1] if typ == "person" else n)
    return entity in listed, wrong


def classify(text, passage_text, entity):
    typ = TYPE[entity]
    nm = named(text, typ)
    narrow = "true" if entity in nm else "other" if nm else "neither"
    true, wrong = claims(text, passage_text, entity)
    return narrow, "true" if true else "other" if wrong else "neither"


if __name__ == "__main__":
    for typ in ENTITIES:
        if typ != "person":
            _matcher(typ)
    wn = {t: _wordnet(t) for t in ROOTS}
    print({t: len(v) for t, v in wn.items()})
    assert {"dog", "zebra", "sparrow"} <= wn["animal"] and "bear" not in wn["animal"] and "animal" not in wn["animal"]
    assert {"bread", "soup", "pizza"} <= wn["food"] and {"cello", "oboe", "trombone"} <= wn["instrument"]
    assert broad_named("They flew from Spain to Tokyo and Madrid.", "country") == {"Spain"}
    assert {"Tokyo", "Madrid"} <= broad_named("They flew from Spain to Tokyo and Madrid.", "city")
    assert broad_named("A zebra and two sparrows", "animal") == {"zebra", "sparrow"}
    p = "He taught piano in Dallas and played the organ."
    assert classify("He taught the piano.", p, "piano") == ("true", "true")
    assert classify("He played the oboe.", p, "piano") == ("neither", "other")       # wrong name outside our 10
    assert classify("He played the violin.", p, "piano") == ("other", "other")
    assert classify("He played the organ.", p, "piano")[1] == "neither"              # organ is in the passage
    assert classify("A painting by Singer Sargent.", "A painting by Picasso.", "Picasso") == ("neither", "other")
    assert classify("A painting by Pablo Picasso.", "A painting by Picasso.", "Picasso") == ("true", "true")
    px = "He played the piano in Dallas."
    ex = "A pianist with a guitar, two violins and the piano. Oboes too."
    assert replace_names(ex, px, "piano", "wrong") == "A pianist with another instrument, two other instruments and the piano. Other instruments too.", replace_names(ex, px, "piano", "wrong")
    assert replace_names(ex, px, "piano", "true") == "A pianist with a guitar, two violins and another instrument. Oboes too."
    assert claims(replace_names(ex, px, "piano", "wrong"), px, "piano") == (True, set())
    assert replace_names("Works by Mark Rothko's circle and Picasso.", "Picasso sold.", "Picasso", "wrong") == "Works by another person's circle and Picasso."
    assert all_names("two violins and Mark Rothko", "instrument") == {"violin"}
    print("ok")
