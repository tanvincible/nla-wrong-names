"""Evaluate a verbalizer checkpoint on held-out test passages (never used in training).

    python evaluate.py released                              the released verbalizer
    python evaluate.py runs/finetune/holdback-holdback/round2/adapter
    options: --passages 540 (default; the seeded step 4 set) or a smaller number, --name <label>

For each test passage at 0, 1, 4 and 16 tokens after the entity: 4 explanations. Reported per distance:
wrong names per explanation, share with at least one wrong name, true-name rate, mean cosine, plus the two
guards (names of any kind per explanation, length). Truth comes from the passage text, never from agreement.
A second block applies the "keep a name only if at least 2 of the 4 samples give it" filter to the same samples.
Writes runs/finetune/eval/<name>.md and .json; generation and reconstruction are cached.
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from entities import boot_mean, cfg_hash, fmt
from fade_curve import DISTS, NLA_CFG, OUT as FADE2, load_acts, load_passages, split
from finetune import HB, reconstruct_batch, sample
from names import all_names, claims
from nla_local import cos, log

N_SAMPLES, EVAL_DISTS = 4, [0, 1, 4, 16]

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--passages", type=int, default=540)
    ap.add_argument("--name", default=None)
    ap.add_argument("--gen_batch", type=int, default=32)
    a = ap.parse_args(argv)
    released = a.checkpoint == "released"
    name = a.name or ("released" if released else "-".join(Path(a.checkpoint).parts[-3:-1]))
    out = HB / "eval"
    out.mkdir(parents=True, exist_ok=True)

    passages, _ = load_passages()
    H, _ = load_acts(passages)
    _, _, test = split(passages)
    by_e = defaultdict(list)
    for i in test:
        by_e[passages[i]["entity"]].append(i)
    chosen = [i for e in by_e for i in by_e[e][:NLA_CFG["per_entity"]]]  # the same 540 as step 4
    if a.passages < len(chosen):
        chosen = sorted(random.Random(0).sample(chosen, a.passages))
    rows = [(i, d) for d in EVAL_DISTS for i in chosen]
    tag = f"{name}-p{len(chosen)}"
    timings = {}

    sf = out / f"{tag}-samples.json"
    if not sf.exists():
        have = {}
        if released:  # reuse the step 4 explanations where they exist
            for f in sorted(FADE2.glob(f"nla-{cfg_hash(NLA_CFG)}*.jsonl")):
                for j in map(json.loads, f.read_text().splitlines()):
                    have[(j["i"], j["d"])] = j["samples"]
        if all(r in have for r in rows):
            log(f"eval {name}: reusing the step 4 explanations for all {len(rows)} activations")
            samples = [have[r] for r in rows]
        else:
            t0 = time.time()
            samples = sample(torch.stack([H[i, DISTS.index(d)] for i, d in rows]), n=N_SAMPLES, batch=a.gen_batch, seed=0,
                             adapter=None if released else a.checkpoint, live=str(out / f"{tag}-live.jsonl"))
            timings["generate"] = round(time.time() - t0, 1)
        sf.write_text(json.dumps(samples))
    samples = json.loads(sf.read_text())
    rf = out / f"{tag}-recs.pt"
    if not rf.exists():
        t0 = time.time()
        torch.save(reconstruct_batch([z for row in samples for z in row]), rf)
        timings["reconstruct"] = round(time.time() - t0, 1)
    recs = torch.load(rf).view(len(rows), N_SAMPLES, -1)
    h = torch.stack([H[i, DISTS.index(d)] for i, d in rows])
    c = cos(recs, h[:, None])

    t0 = time.time()
    res = {"name": name, "passages": len(chosen), "per_distance": {}, "timings_seconds": timings}
    L = [f"# {name}: {len(chosen)} held-out test passages x {len(EVAL_DISTS)} distances x {N_SAMPLES} samples (natural text)\n",
         "| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate | mean cosine | "
         "any same-type names per explanation | length (chars) |", "|---|---|---|---|---|---|---|"]
    F = ["\nWith the filter \"keep a name only if at least 2 of the 4 samples give it\" applied to the same samples:\n",
         "| tokens after entity | wrong names per explanation | share with at least one wrong name | true-name rate |", "|---|---|---|---|"]
    for d in EVAL_DISTS:
        m = defaultdict(lambda: defaultdict(list))  # measure -> entity -> values
        for r, (i, dd) in enumerate(rows):
            if dd != d:
                continue
            p = passages[i]
            cl = [claims(z, p["text"], p["entity"]) for z in samples[r]]
            cnt = Counter(n for t, w in cl for n in ([p["entity"]] if t else []) + sorted(w))
            for s, (t, w) in enumerate(cl):
                e = p["entity"]
                m["wrong"][e].append(len(w)); m["any_wrong"][e].append(bool(w)); m["true"][e].append(t)
                m["cos"][e].append(c[r, s].item()); m["names"][e].append(len(all_names(samples[r][s], p["type"])))
                m["len"][e].append(len(samples[r][s]))
                kept = [n for n in w if cnt[n] >= 2]
                m["f_wrong"][e].append(len(kept)); m["f_any_wrong"][e].append(bool(kept)); m["f_true"][e].append(t and cnt[e] >= 2)
        ci = {k: boot_mean(v) for k, v in m.items()}
        res["per_distance"][d] = {k: list(map(float, v)) for k, v in ci.items()}
        L.append(f"| {d} | {fmt(ci['wrong'])} | {fmt(ci['any_wrong'])} | {fmt(ci['true'])} | {fmt(ci['cos'])} | "
                 f"{fmt(ci['names'])} | {ci['len'][0]:.0f} |")
        F.append(f"| {d} | {fmt(ci['f_wrong'])} | {fmt(ci['f_any_wrong'])} | {fmt(ci['f_true'])} |")
    timings["score_names"] = round(time.time() - t0, 1)
    L += F + [f"\nSeconds per stage: {timings or 'all cached'}. Intervals: bootstrap over entities (10k resamples)."]
    (out / f"{tag}.md").write_text("\n".join(L))
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
