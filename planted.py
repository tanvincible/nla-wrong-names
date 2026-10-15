"""Planted concepts: add a known concept vector to a natural activation; does the verbalizer name it?

Conditions:  planted  h + s*|h|*unit(concept vector)
             random   h + s*|h|*unit(gaussian)          control
             none     h                                  control
             natural  text that actually mentions the concept (positive control)
Scoring is judge-free word matching. Every stage caches under runs/planted/, so rerunning resumes
and then reprints the report.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from nla_local import extract, verbalize

OUT = Path(__file__).parent / "runs" / "planted"
N = 4  # samples per activation
STRENGTHS = [0.5, 1.0, 2.0]  # norm of the added vector as a multiple of |h|

# name -> (type, phrase used in prompts, strings that count as naming it)
CONCEPTS = {
    "Baudelaire": ("poet", "Baudelaire", ["baudelaire"]),
    "Verlaine": ("poet", "Verlaine", ["verlaine"]),
    "Shakespeare": ("poet", "Shakespeare", ["shakespeare"]),
    "Whitman": ("poet", "Walt Whitman", ["whitman"]),
    "Paris": ("city", "Paris", ["paris"]),
    "Tokyo": ("city", "Tokyo", ["tokyo"]),
    "Cairo": ("city", "Cairo", ["cairo"]),
    "Chicago": ("city", "Chicago", ["chicago"]),
    "rabbit": ("animal", "rabbits", ["rabbit", "bunny", "bunnies"]),
    "mouse": ("animal", "mice", ["mouse", "mice"]),
    "elephant": ("animal", "elephants", ["elephant"]),
    "dolphin": ("animal", "dolphins", ["dolphin"]),
}
# baseline for concept vectors: mean activation over unrelated words (Lindsey 2025 recipe)
BASELINE = ("chairs bridges music rivers engines soup taxes mountains glass sleep football paper clouds "
            "medicine windows friendship salt clocks farming electricity shoes languages doors gold").split()
CARRIERS = [
    "A good sourdough loaf starts the night before. Mix the flour and water and let them rest for an hour, "
    "then fold in the starter and the salt. Over the next four hours, stretch and fold the dough every "
    "thirty minutes until it feels smooth and elastic. Shape it gently, place it in a floured basket, and "
    "leave it in the refrigerator overnight. In the morning, heat the oven as high as it will go and",
    "This release focuses on stability and performance. We fixed a crash that occurred when importing "
    "large files, reduced memory use during background sync by roughly a third, and rewrote the search "
    "index so that results appear as you type. Several settings pages were reorganized to make common "
    "options easier to find. If you run into problems after updating, restart the application and",
]
NATURAL = " Anyway, that reminds me: I have been reading a great deal about {} lately, and"


def unit(v):
    return v / v.norm()


def build():
    """Run the target model once; return rows (metadata) and their activation vectors."""
    names = list(CONCEPTS)
    prompts = [f"Tell me about {CONCEPTS[c][1]}." for c in names] + [f"Tell me about {w}." for w in BASELINE]
    pa, _ = extract(prompts, chat=True)
    last = torch.stack([a[-1] for a in pa])
    cvec = {c: last[i] - last[len(names):].mean(0) for i, c in enumerate(names)}

    ta, _ = extract(CARRIERS + [CARRIERS[0] + NATURAL.format(CONCEPTS[c][1]) for c in names])
    hs = [a[-1] for a in ta[:len(CARRIERS)]]
    g = torch.Generator().manual_seed(0)

    rows, vecs = [], []
    def add(vec, **meta):
        rows.append({"id": len(rows), **meta})
        vecs.append(vec)

    for k, h in enumerate(hs):
        add(h, cond="none", carrier=k)
        for s in STRENGTHS:
            for c in names:
                add(h + s * h.norm() * unit(cvec[c]), cond="planted", carrier=k, strength=s, concept=c)
            for _ in range(2):
                add(h + s * h.norm() * unit(torch.randn(h.shape, generator=g)), cond="random", carrier=k, strength=s)
    for i, c in enumerate(names):
        add(ta[len(CARRIERS) + i][-1], cond="natural", concept=c)

    r = next(r for r in rows if r["cond"] == "planted")
    d = vecs[r["id"]] - hs[r["carrier"]]
    assert torch.cosine_similarity(d, cvec[r["concept"]], dim=0) > 0.999 and d.norm() > 0
    return rows, torch.stack(vecs), cvec


def named(text):
    t = text.lower()
    return {c for c, (_, _, keys) in CONCEPTS.items() if any(k in t for k in keys)}


def report(rows, exps):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        hits = [named(e) for e in exps[r["id"]]]
        key = (r["cond"], r.get("strength", ""))
        if "concept" in r:
            c = r["concept"]
            rivals = {x for x in CONCEPTS if x != c and CONCEPTS[x][0] == CONCEPTS[c][0]}
            agg[key]["named (per sample)"] += [c in h for h in hits]
            agg[key]["named (any of %d)" % N].append(any(c in h for h in hits))
            agg[key]["same-type rival named"] += [bool(h & rivals) for h in hits]
        else:
            agg[key]["any concept named (false alarm)"] += [bool(h) for h in hits]
    print(f"\n{'condition':<10}{'strength':<10}{'measure':<34}{'rate':>6}{'n':>6}")
    for (cond, s), ms in sorted(agg.items(), key=str):
        for m, v in ms.items():
            print(f"{cond:<10}{str(s):<10}{m:<34}{sum(v) / len(v):>6.2f}{len(v):>6}")
    print("\nplanted, strength 1.0, named per sample by concept:")
    for c in CONCEPTS:
        v = [c in named(e) for r in rows if r["cond"] == "planted" and r["concept"] == c and r["strength"] == 1.0
             for e in exps[r["id"]]]
        print(f"  {c:<12}{sum(v)}/{len(v)}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    acts, log = OUT / "acts.pt", OUT / "explanations.jsonl"
    if not acts.exists():
        rows, vecs, cvec = build()
        torch.save({"rows": rows, "vecs": vecs, "cvec": cvec}, acts)
    d = torch.load(acts)
    rows, vecs = d["rows"], d["vecs"]

    exps = {j["id"]: j["samples"] for j in map(json.loads, log.read_text().splitlines())} if log.exists() else {}
    todo = [r for r in rows if r["id"] not in exps]
    for i in range(0, len(todo), 24):  # one model load per chunk; results saved after each
        chunk, t = todo[i:i + 24], time.time()
        out = verbalize(torch.stack([vecs[r["id"]] for r in chunk]), n=N)
        with log.open("a") as f:
            for r, e in zip(chunk, out):
                exps[r["id"]] = e
                f.write(json.dumps({**r, "samples": e}) + "\n")
        print(f"{len(exps)}/{len(rows)} activations done, {(time.time() - t) / (len(chunk) * N):.1f}s per explanation",
              flush=True)
    report(rows, exps)
