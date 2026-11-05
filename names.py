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
"""
import functools
import re

from entities import ENTITIES, PROPER, TYPE, forms, named

GENERIC = {  # category words, not names of a specific thing
    "animal": "animal creature beast pet male female young adult predator prey game stock livestock mate giant "
              "feeder racer stray survivor pest critter fauna brute varmint darter".split(),
    "food": "food dish meal course fare snack treat feed produce drink beverage nutrient nutriment diet ration meat "
            "fruit vegetable seafood dessert sweet staple ingredient refreshment delicacy leftovers serving helping "
            "portion plate side special mess mix stuff liquid water pop spread cut joint side slice piece".split(),
    "instrument": "instrument keyboard string strings wind brass percussion woodwind section key keys pipe bones".split(),
}
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
    """-> function text -> set of names of this type found in the text."""
    if typ == "person":
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        return lambda text: {e.text.strip() for e in nlp(text).ents if e.label_ == "PERSON"}
    if typ in ("country", "city"):
        import geonamescache
        gc = geonamescache.GeonamesCache()
        countries = {c["name"] for c in gc.get_countries().values()} | set(EXTRA_COUNTRIES)
        if typ == "country":
            names = countries
        else:
            words = {w.strip() for w in open("/usr/share/dict/words")}  # lowercase entries = ordinary words
            names = {c["name"] for c in gc.get_cities().values()
                     if c["population"] >= 300_000 and c["name"].lower() not in words and c["name"] not in countries}
        names = {n for n in names if re.fullmatch(r"[A-Z][\w .'\-]+", n)} | set(ENTITIES[typ])
        pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")\b")
        return lambda text: set(pat.findall(text))
    names = _wordnet(typ) | {f.lower() for e in ENTITIES[typ] for f in forms(e)}
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")(?:e?s)?\b", re.I)
    return lambda text: {m.lower() for m in pat.findall(text)}


def broad_named(text, typ):
    return _matcher(typ)(text)


def classify(text, passage_text, entity):
    typ = TYPE[entity]
    nm = named(text, typ)
    narrow = "true" if entity in nm else "other" if nm else "neither"
    if entity in nm:
        return narrow, "true"
    low = passage_text if typ in PROPER else passage_text.lower()
    mine = {f if typ in PROPER else f.lower() for f in forms(entity)}

    def wrong(n):  # a same-type name that is neither the true entity nor anything the passage says
        parts = [n] + (n.split() if typ == "person" else [])
        return not any(p in mine for p in parts) and not any(re.search(r"\b" + re.escape(p), low) for p in parts)
    return narrow, "other" if any(wrong(n) for n in broad_named(text, typ)) else "neither"


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
    print("ok")
