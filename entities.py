"""60 entities (10 each of 6 types), whole-word matching, config hashing and interval helpers."""
import hashlib
import json
import math
import random
import re

PROPER = {"country", "city", "person"}
ENTITIES = {
    "country": "France Japan Egypt Brazil Canada India Germany Italy Mexico Russia".split(),
    "city": "Paris Tokyo Cairo Chicago London Berlin Moscow Sydney Toronto Rome".split(),
    "person": "Shakespeare Napoleon Einstein Mozart Darwin Lincoln Churchill Gandhi Picasso Beethoven".split(),
    "animal": "dog cat horse lion tiger elephant wolf eagle rabbit dolphin".split(),
    "food": "bread cheese rice pasta chocolate pizza butter honey potato sugar".split(),
    "instrument": "piano guitar violin cello flute trumpet saxophone clarinet harp banjo".split(),
}
# fixed synonyms: extra surface forms that count as naming the entity
SYNONYMS = {"wolf": ["wolves"], "potato": ["potatoes"], "rabbit": ["bunny", "bunnies"], "dog": ["puppy", "puppies"],
            "cat": ["kitten", "kittens"], "bread": ["loaf", "loaves"]}
TYPE = {e: t for t, es in ENTITIES.items() for e in es}


def forms(e):
    return [e] + ([] if TYPE[e] in PROPER else [e + "s"]) + SYNONYMS.get(e, [])


def phrase(e):
    """how the entity is written in 'Tell me about {x}.'"""
    return e if TYPE[e] in PROPER or TYPE[e] == "food" else forms(e)[1]


# proper nouns match case-sensitively; common nouns in any case
_PAT = {e: re.compile(r"\b(" + "|".join(forms(e)) + r")\b", 0 if TYPE[e] in PROPER else re.I) for e in TYPE}


def mentions(text, e):
    return list(_PAT[e].finditer(text))


def named(text, typ):
    """entities of this type named in the text (whole-word)."""
    return {e for e in ENTITIES[typ] if _PAT[e].search(text)}


def swap(text, c, x):
    """Replace every mention of entity c with same-type entity x (plural and synonym forms -> x's plural)."""
    def f(m):
        out = x if m.group(1) == c or TYPE[x] in PROPER else forms(x)[1]
        return out.capitalize() if m.group(1)[0].isupper() and TYPE[x] not in PROPER else out
    return _PAT[c].sub(f, text)


def cfg_hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def wilson(k, n):
    """Wilson 95% interval for a plain proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    z, p = 1.96, k / n
    mid, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, mid - half, mid + half


def boot_mean(by_entity, n=10000, seed=0):
    """Mean and 95% interval, resampling entities (fast path of boot_entities for means)."""
    import numpy as np
    ks = [k for k in by_entity if len(by_entity[k])]
    s = np.array([sum(by_entity[k]) for k in ks], float)
    c = np.array([len(by_entity[k]) for k in ks], float)
    idx = np.random.default_rng(seed).integers(0, len(ks), (n, len(ks)))
    lo, hi = np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])
    return s.sum() / c.sum(), lo, hi


def fmt(r):
    return f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"


def boot_entities(by_entity, stat=lambda xs: sum(xs) / len(xs), n=10000, seed=0):
    """Point estimate and 95% interval, resampling entities. by_entity: {entity: [values]}."""
    keys = [k for k in by_entity if by_entity[k]]
    flat = lambda ks: [v for k in ks for v in by_entity[k]]
    rng = random.Random(seed)
    ms = sorted(stat(flat(rng.choices(keys, k=len(keys)))) for _ in range(n))
    return stat(flat(keys)), ms[int(.025 * n)], ms[int(.975 * n)]


if __name__ == "__main__":
    assert len(TYPE) == 60 and all(len(v) == 10 for v in ENTITIES.values())
    assert named("Two wolves and a Cat near paris", "animal") == {"wolf", "cat"}
    assert named("paris", "city") == set() and named("Parisian Paris.", "city") == {"Paris"}
    assert phrase("dog") == "dogs" and phrase("bread") == "bread" and phrase("Paris") == "Paris"
    p, lo, hi = wilson(5, 10)
    assert abs(p - .5) < 1e-9 and 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    m, lo, hi = boot_mean({"a": [1, 1, 0], "b": [0, 0], "c": [1]})
    assert abs(m - 0.5) < 1e-9 and lo < m < hi
    assert swap("A wolf, two Wolves, Paris", "wolf", "cat") == "A cat, two Cats, Paris"
    assert swap("In Paris.", "Paris", "Rome") == "In Rome."
    print("ok")
