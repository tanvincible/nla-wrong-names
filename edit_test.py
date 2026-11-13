"""Does the reconstruction score pay for wrong names? A check on existing explanations (natural text).

Take step 4 explanations at distances 0 and 1 that name the true entity AND give at least one wrong same-type
name in the part describing the context (cleaned detector). Make two edited copies of each:
  (a) every wrong same-type name replaced by "another <type>"      (wrong names removed)
  (b) control: every mention of the TRUE name replaced the same way (wrong names left in)
Wrong names are replaced wherever they occur in the explanation, not only in the context part.
Reconstruct all three and compare cosine with the real activation. Intervals: bootstrap over passages (10k).
Writes runs/analysis/edit_test.md; reconstructions cached in runs/analysis/edit_test-recs.pt.
"""
import json
import random
from collections import defaultdict

import torch

from entities import boot_mean, cfg_hash, fmt
from fade_curve import DISTS, NLA_CFG, OUT, load_acts, load_passages
from names import claims, replace_names
from nla_local import cos, reconstruct
from name_analysis import parts

MAX_N, SEED = 1000, 0

if __name__ == "__main__":
    passages, _ = load_passages()
    H, _ = load_acts(passages)
    done = {}
    for f in sorted(OUT.glob(f"nla-{cfg_hash(NLA_CFG)}*.jsonl")):
        for j in map(json.loads, f.read_text().splitlines()):
            done[(j["i"], j["d"])] = j["samples"]

    pool = []
    for (i, d), samples in sorted(done.items()):
        if d not in (0, 1):
            continue
        p = passages[i]
        for s in samples:
            true, wrong = claims(s, p["text"], p["entity"])
            if true and claims(parts(s)[0], p["text"], p["entity"])[1]:
                pool.append((i, d, s, len(wrong)))
    chosen = random.Random(SEED).sample(pool, min(MAX_N, len(pool)))
    rows = []
    for i, d, s, n_wrong in chosen:
        p = passages[i]
        a, b = replace_names(s, p["text"], p["entity"], "wrong"), replace_names(s, p["text"], p["entity"], "true")
        rows.append(dict(i=i, d=d, orig=s, a=a, b=b, n_wrong=n_wrong,
                         a_clean=not claims(a, p["text"], p["entity"])[1], a_true=claims(a, p["text"], p["entity"])[0],
                         b_gone=not claims(b, p["text"], p["entity"])[0]))
    print(f"eligible explanations: {len(pool)} (distance 0: {sum(r[1] == 0 for r in pool)}, distance 1: "
          f"{sum(r[1] == 1 for r in pool)}); sampled {len(rows)}", flush=True)

    cache = OUT / "edit_test-recs.pt"
    recs = torch.load(cache) if cache.exists() else {}
    todo = sorted({r[k] for r in rows for k in ("orig", "a", "b")} - recs.keys())
    if todo:
        recs.update(zip(todo, reconstruct(todo)))
        torch.save(recs, cache)
    for r in rows:
        h = H[r["i"], DISTS.index(r["d"])]
        r.update({"c_" + k: cos(recs[r[k]], h).item() for k in ("orig", "a", "b")})

    L = ["Natural text. Explanations that name the true entity and give at least one wrong same-type name in the context "
         f"part. {len(pool)} eligible at distances 0 and 1; {len(rows)} sampled (seed {SEED}). Intervals: bootstrap over passages.\n",
         f"Edit check: after (a) no wrong name is detected in {sum(r['a_clean'] for r in rows) / len(rows):.1%} of copies and the true "
         f"name is still there in {sum(r['a_true'] for r in rows) / len(rows):.1%}; after (b) the true name is gone in "
         f"{sum(r['b_gone'] for r in rows) / len(rows):.1%}.\n",
         "| tokens after entity | n | cosine, original | cosine, (a) wrong names removed | cosine, (b) true name removed | "
         "original beats (a) | original beats (b) | gap original - (a) | gap original - (b) | gap per wrong name removed |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for label, ds in (("0", [0]), ("1", [1]), ("0 and 1 pooled", [0, 1])):
        sub = [r for r in rows if r["d"] in ds]

        def by(fn):
            out = defaultdict(list)
            for r in sub:
                out[r["i"]].append(fn(r))
            return fmt(boot_mean(out))
        L.append(f"| {label} | {len(sub)} | " + " | ".join([
            by(lambda r: r["c_orig"]), by(lambda r: r["c_a"]), by(lambda r: r["c_b"]),
            by(lambda r: r["c_orig"] > r["c_a"]), by(lambda r: r["c_orig"] > r["c_b"]),
            by(lambda r: r["c_orig"] - r["c_a"]), by(lambda r: r["c_orig"] - r["c_b"]),
            by(lambda r: (r["c_orig"] - r["c_a"]) / r["n_wrong"])]) + " |")
    ex = random.Random(1).sample(rows, 3)
    L.append("\nThree examples of the edits:\n")
    for r in ex:
        L += [f"- original ({r['c_orig']:.4f}): {r['orig'][:260]!r}", f"  - (a) ({r['c_a']:.4f}): {r['a'][:260]!r}",
              f"  - (b) ({r['c_b']:.4f}): {r['b'][:260]!r}"]
    (OUT / "edit_test.md").write_text("\n".join(L))
    print("\n".join(L))
