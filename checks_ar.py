"""Reader-only cheap checks on the planted-concepts explanations (run planted.py first).

  rivals  one-claim rivals: does the true explanation beat the same text with the name swapped for a rival?
  G       per-claim gain: score with the claim minus score with it hedged, true vs false claims
  C       near-twin lineup: does the rebuild pick its own activation over twins differing in one concept?
  E       consensus: are claims repeated across samples more often true?  (no model needed)

Ground truth comes from planting: a claim is true iff it names the concept planted in (or mentioned in
the text behind) that activation. Reconstructions are cached in runs/checks/recs.pt.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch

from nla_local import cos, reconstruct
from planted import CONCEPTS, OUT as PLANTED, named

OUT = Path(__file__).parent / "runs" / "checks"
HEDGE = {"poet": ("a poet", "poets"), "city": ("a city", "cities"), "animal": ("an animal", "animals")}
PLURALS = {"mice", "bunnies"}


def edit(text, c, to=None):
    """Replace every mention of concept c with concept `to`, or with a type-level hedge if to is None."""
    typ = CONCEPTS[c][0]
    pat = re.compile(r"\b(" + "|".join(sorted(CONCEPTS[c][2], key=len, reverse=True)) + r")(s?)\b", re.I)

    def f(m):
        plural = bool(m.group(2)) or m.group(1).lower() in PLURALS
        if to is None:
            return HEDGE[typ][plural]
        return CONCEPTS[to][1] if plural and typ == "animal" else to
    return pat.sub(f, text)


def rivals(c):
    return [x for x in CONCEPTS if x != c and CONCEPTS[x][0] == CONCEPTS[c][0]]


def boot(xs, n=2000):
    """mean and bootstrap 95% interval over items."""
    if not xs:
        return "n=0"
    xs = [float(x) for x in xs]
    rng = random.Random(0)
    ms = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return f"{sum(xs) / len(xs):+.3f} [{ms[int(.025 * n)]:+.3f}, {ms[int(.975 * n)]:+.3f}] n={len(xs)}"


def load():
    d = torch.load(PLANTED / "acts.pt")
    rows = [json.loads(l) for l in (PLANTED / "explanations.jsonl").read_text().splitlines()]
    return rows, d["vecs"]


def claims(rows):
    """Every (row, sample, concept named) with whether the claim is true."""
    for r in rows:
        for e in r["samples"]:
            for c in named(e):
                yield r, e, c, c == r.get("concept")


if __name__ == "__main__":
    assert edit("Two rabbits and Baudelaire's rabbit", "rabbit", "mouse") == "Two mice and Baudelaire's mouse"
    assert edit("In Paris, rabbits", "rabbit") == "In Paris, animals" and named(edit("In Paris", "Paris")) == set()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, vecs = load()

    texts = {e for r in rows for e in r["samples"]}
    for r, e, c, true in claims(rows):
        texts.add(edit(e, c))
        if true:
            texts |= {edit(e, c, x) for x in rivals(c)}
    cache = OUT / "recs.pt"
    recs = torch.load(cache) if cache.exists() else {}
    todo = sorted(texts - recs.keys())
    if todo:
        print(f"reconstructing {len(todo)} texts")
        recs.update(zip(todo, reconstruct(todo)))
        torch.save(recs, cache)
    score = lambda text, r: cos(recs[text], vecs[r["id"]]).item()

    print("\n== one-claim rivals: true explanation vs same text with the name swapped for a same-type rival")
    wins, margins = [], []
    for r, e, c, true in claims(rows):
        if true:
            for x in rivals(c):
                wins.append(score(e, r) > score(edit(e, c, x), r))
                margins.append(score(e, r) - score(edit(e, c, x), r))
    print(f"true beats rival (rate):   {boot(wins)}   (published baseline 0.602)")
    print(f"cosine margin true-rival:  {boot(margins)}")

    print("\n== G: per-claim gain = cos(with claim) - cos(claim hedged to its type)")
    gain = defaultdict(list)
    for r, e, c, true in claims(rows):
        gain[true].append(score(e, r) - score(edit(e, c), r))
    print(f"true claims:   {boot(gain[True])}   positive: {boot([g > 0 for g in gain[True]])}")
    print(f"false claims:  {boot(gain[False])}   positive: {boot([g > 0 for g in gain[False]])}")

    print("\n== C: near-twin lineup (4 same-type activations, chance 0.25): does the rebuild pick its own?")
    line = defaultdict(list)
    for r in rows:
        if "concept" not in r:
            continue
        same = lambda q: q["cond"] == r["cond"] and q.get("carrier") == r.get("carrier") and q.get("strength") == r.get("strength")
        twins = [q for q in rows if same(q) and q["concept"] in rivals(r["concept"])]
        assert len(twins) == 3
        for e in r["samples"]:
            win = all(score(e, r) > score(e, q) for q in twins)
            line[(r["cond"], r.get("strength", ""), "names it" if r["concept"] in named(e) else "does not name it")].append(win)
    for k in sorted(line, key=str):
        print(f"{k[0]:<9}{str(k[1]):<5}{k[2]:<18}{boot(line[k])}")

    print("\n== E: consensus over the 4 samples of one activation: share of claims that are true")
    prec = defaultdict(list)
    for r in rows:
        counts = defaultdict(int)
        for e in r["samples"]:
            for c in named(e):
                counts[c] += 1
        for c, k in counts.items():
            prec[k].append(c == r.get("concept"))
    for k in sorted(prec):
        print(f"claim made in {k} of 4 samples:  {boot(prec[k])}")
