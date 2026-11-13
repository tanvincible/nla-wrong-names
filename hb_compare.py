"""Paired comparison of evaluated checkpoints with the released verbalizer (natural text, held-out passages).

    python hb_compare.py full-recon-round2 full-holdback-round2 full-lineup-round2
    python hb_compare.py <name> --holdout runs/hb_pod/<run>/holdout.json   also split by entities held out of training

Reads the explanations hb_eval.py saved (runs/hb_pod/eval or runs/hb/eval). Every checkpoint explained the
same 540 passages at the same distances, so the difference is taken per entity (checkpoint minus released)
and the interval is a bootstrap over entities (10k). Writes runs/hb/compare.md.
"""
import json
import sys
from collections import defaultdict

from entities import boot_mean
from fade2 import NLA_CFG, load_passages, split
from hb_eval import EVAL_DISTS
from names import claims
from nla_local import RUNS

EVAL = next(d for d in (RUNS / "hb_pod" / "eval", RUNS / "hb" / "eval") if d.exists())


def per_entity(name, passages, rows):
    """-> {distance: {entity: (wrong names per explanation, true-name rate)}}"""
    samples = json.loads((EVAL / f"{name}-p540-samples.json").read_text())
    assert len(samples) == len(rows), (name, len(samples), len(rows))
    acc = defaultdict(lambda: defaultdict(list))
    for (i, d), row in zip(rows, samples):
        p = passages[i]
        for z in row:
            t, w = claims(z, p["text"], p["entity"])
            acc[d][p["entity"]].append((len(w), t))
    return {d: {e: (sum(w for w, _ in v) / len(v), sum(t for _, t in v) / len(v)) for e, v in by.items()} for d, by in acc.items()}


if __name__ == "__main__":
    passages, _ = load_passages()
    _, _, test = split(passages)
    by_e = defaultdict(list)
    for i in test:
        by_e[passages[i]["entity"]].append(i)
    chosen = [i for e in by_e for i in by_e[e][:NLA_CFG["per_entity"]]]  # the same 540 as hb_eval.py
    rows = [(i, d) for d in EVAL_DISTS for i in chosen]
    base = per_entity("released", passages, rows)
    L = ["Natural text. Each checkpoint minus the released verbalizer on the same 540 held-out passages, 4 samples each. "
         "Intervals: bootstrap over entities (10k). An interval that excludes 0 is a real difference.\n",
         "| checkpoint | tokens after entity | wrong names per explanation, released | difference | relative | true-name rate, released | difference |",
         "|---|---|---|---|---|---|---|"]
    args = sys.argv[1:]
    held = json.loads(open(args.pop(args.index("--holdout") + 1)).read()) if "--holdout" in args else []
    args = [a for a in args if a != "--holdout"]
    groups = [("", lambda e: True)] + ([(" (entities trained on)", lambda e: e not in held),
                                        (" (entities held out of training)", lambda e: e in held)] if held else [])
    cache = {}
    for name, (glabel, member) in ((n, g) for n in args for g in groups):
        arm = cache[name] = cache.get(name) or per_entity(name, passages, rows)
        for label, ds in [(str(d), [d]) for d in EVAL_DISTS] + [("all four", EVAL_DISTS)]:
            ents, name_ = sorted(filter(member, base[ds[0]])), name + glabel
            mean = lambda src, k, e: sum(src[d][e][k] for d in ds) / len(ds)
            bw = sum(mean(base, 0, e) for e in ents) / len(ents)
            bt = sum(mean(base, 1, e) for e in ents) / len(ents)
            dw = boot_mean({e: [mean(arm, 0, e) - mean(base, 0, e)] for e in ents})
            dt = boot_mean({e: [mean(arm, 1, e) - mean(base, 1, e)] for e in ents})
            L.append(f"| {name_} | {label} | {bw:.3f} | {dw[0]:+.3f} [{dw[1]:+.3f}, {dw[2]:+.3f}] | {dw[0] / bw:+.1%} | "
                     f"{bt:.1%} | {dt[0]:+.1%} [{dt[1]:+.1%}, {dt[2]:+.1%}] |")
    (RUNS / "hb").mkdir(parents=True, exist_ok=True)
    (RUNS / "hb" / ("compare-holdout.md" if held else "compare.md")).write_text("\n".join(L))
    print("\n".join(L))
