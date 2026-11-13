"""What the 8,640 saved explanations name, and whether their samples agree (no new generation). Natural text.

  A  the three-bin table (true / wrong same-type / neither) with the current name detector
  B  non-exclusive rate: explanations containing at least one wrong same-type name, whatever else they say
  C  fair probe comparison: the probe on its most confident N% of the same 540 passages
  D  consensus: precision of a name by how many of the 4 samples give it, and the AUC of that count
  E  filter trade-off for "keep names given by at least k of 4"
  F  where names sit: the part that describes the context vs the part that predicts the continuation
  G  what the wrong names are at distances 4 and 16, with 15 random examples each

The name detector is names.py; NAMES_MODE=clean (default) or raw. Writes runs/analysis/name_analysis.md.
Intervals: bootstrap over entities (10k resamples).
"""
import json
import os
import random
import re
from collections import Counter, defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from entities import ENTITIES, TYPE, boot_mean, cfg_hash, fmt
from fade_curve import CS, DISTS, NLA_CFG, OUT, load_acts, load_passages, split
from names import claims

MODE = os.environ.get("NAMES_MODE", "clean")
# Every explanation has three paragraphs; the third starts 'Final token "..."' and, after one of these cues,
# lists what the verbalizer expects next. Hand-checked on 50: 48 split cleanly, 1 had no cue, 1 split a clause early.
CUE = re.compile(r"\b(strongly expecting|expecting|strongly requiring|requiring|likely continuing|likely to continue|"
                 r"continuing with|suggesting continuation|to complete|completing|awaiting|anticipating|expects|"
                 r"likely followed by|followed by|needs|needing|strongly suggesting|suggesting)\b", re.I)


def parts(text):
    """-> (the part describing the context, the part predicting the continuation)."""
    k = text.rfind("Final token")
    m = CUE.search(text, k) if k >= 0 else None
    return (text[:m.start()], text[m.start():]) if m else (text, "")


def pct(by):
    return fmt(boot_mean(by)) if any(by.values()) else "n/a"


if __name__ == "__main__":
    passages, cap = load_passages()
    H, cv = load_acts(passages)
    fit, val, test = split(passages)
    train, ent, X = fit + val, [p["entity"] for p in passages], H.numpy()
    done = {}
    for f in sorted(OUT.glob(f"nla-{cfg_hash(NLA_CFG)}*.jsonl")):
        for j in map(json.loads, f.read_text().splitlines()):
            done[(j["i"], j["d"])] = j["samples"]
    dists = NLA_CFG["dists"]
    chosen = sorted({i for i, _ in done})
    n_exp = sum(map(len, done.values()))
    L = [f"All natural text. {len(chosen)} test passages, {n_exp} explanations, 4 per activation. Name detector: {MODE}.\n"]

    # one record per explanation
    recs = defaultdict(list)  # d -> [dict]
    per = {}  # (i, d) -> [(name, samples giving it, is true)]
    for (i, d), samples in done.items():
        cnt, e, ptext = Counter(), ent[i], passages[i]["text"]
        for s in samples:
            true, wrong = claims(s, ptext, e)
            ctx, cont = parts(s)
            tc, wc = claims(ctx, ptext, e)
            tn, wn = claims(cont, ptext, e) if cont else (False, set())
            recs[d].append(dict(i=i, e=e, text=s, true=true, wrong=wrong, tc=tc, wc=wc, tn=tn, wn=wn))
            cnt.update(([e] if true else []) + sorted(wrong))
        per[(i, d)] = [(n, k, n == e) for n, k in cnt.items()]

    def by_entity(d, fn):
        out = defaultdict(list)
        for r in recs[d]:
            out[r["e"]] += fn(r)
        return out

    L += ["### A. What the verbalizer names (three exclusive bins)\n",
          "| tokens after entity | names the true entity | names only wrong same-type entities | names neither | counts |", "|---|---|---|---|---|"]
    for d in dists:
        b = [("true", lambda r: [r["true"]]), ("wrong", lambda r: [not r["true"] and bool(r["wrong"])]),
             ("neither", lambda r: [not r["true"] and not r["wrong"]])]
        c = [sum(fn(r)[0] for r in recs[d]) for _, fn in b]
        L.append(f"| {d} | " + " | ".join(pct(by_entity(d, fn)) for _, fn in b) + f" | {c[0]} / {c[1]} / {c[2]} |")

    L += ["\n### B. Non-exclusive rate: explanations with at least one wrong same-type name\n",
          "Counted whether or not the true entity is also named.\n",
          "| tokens after entity | at least one wrong name | names the true entity | both | wrong names per explanation |", "|---|---|---|---|---|"]
    for d in dists:
        L.append(f"| {d} | {pct(by_entity(d, lambda r: [bool(r['wrong'])]))} | {pct(by_entity(d, lambda r: [r['true']]))} | "
                 f"{pct(by_entity(d, lambda r: [r['true'] and bool(r['wrong'])]))} | "
                 f"{sum(len(r['wrong']) for r in recs[d]) / len(recs[d]):.2f} |")

    L += ["\n### C. Fair probe comparison: the probe on its most confident passages\n",
          "The probe is scored on its most confident N% of the same 540 passages, N being the share of explanations in "
          "which the verbalizer names anything of the right type. Confidence = the probe's top predicted probability.\n",
          "| tokens after entity | N (verbalizer names something) | probe, all 540 passages | probe, most confident N% | verbalizer precision when it names |",
          "|---|---|---|---|---|"]
    for d in dists:
        k = DISTS.index(d)
        conf, ok = {}, {}
        for typ in ENTITIES:
            sel = lambda idx: [i for i in idx if TYPE[ent[i]] == typ]
            f_, v_, tr, te = sel(fit), sel(val), sel(train), sel(chosen)
            sc = StandardScaler().fit(X[f_, k])
            acc = {C: LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[f_, k]), [ent[i] for i in f_])
                   .score(sc.transform(X[v_, k]), [ent[i] for i in v_]) for C in CS}
            best = max(CS, key=lambda C: (acc[C], -C))
            sc = StandardScaler().fit(X[tr, k])
            clf = LogisticRegression(C=best, max_iter=3000).fit(sc.transform(X[tr, k]), [ent[i] for i in tr])
            for i, row in zip(te, clf.predict_proba(sc.transform(X[te, k]))):
                conf[i], ok[i] = row.max(), clf.classes_[row.argmax()] == ent[i]
        said = [r["true"] or bool(r["wrong"]) for r in recs[d]]
        n_frac = sum(said) / len(said)
        top = sorted(conf, key=conf.get, reverse=True)[:round(n_frac * len(conf))]
        group = lambda idx: {e: [ok[i] for i in idx if ent[i] == e] for e in sorted({ent[i] for i in idx})}
        prec = by_entity(d, lambda r: [r["true"]] if r["true"] or r["wrong"] else [])
        L.append(f"| {d} | {n_frac:.1%} | {fmt(boot_mean(group(list(conf))))} | {fmt(boot_mean(group(top)))} (n={len(top)}) | "
                 f"{pct(prec)} (n={sum(map(len, prec.values()))}) |")

    def names_at(ds, keep=lambda c: True):
        """{entity: [is_true]} over every name given at these distances whose sample count passes `keep`."""
        out = defaultdict(list)
        for (i, dd), names in per.items():
            if dd in ds:
                out[ent[i]] += [t for _, c, t in names if keep(c)]
        return out

    def auc_ci(ds):
        by = defaultdict(list)
        for (i, dd), names in per.items():
            if dd in ds:
                by[ent[i]] += [(t, c) for _, c, t in names]
        auc = lambda xs: roc_auc_score([y for y, _ in xs], [s for _, s in xs]) if 0 < sum(y for y, _ in xs) < len(xs) else float("nan")
        keys, rng = list(by), random.Random(0)
        ms = sorted(m for m in (auc([v for k in rng.choices(keys, k=len(keys)) for v in by[k]]) for _ in range(10000)) if m == m)
        return fmt((auc([v for k in keys for v in by[k]]), ms[int(.025 * len(ms))], ms[int(.975 * len(ms)) - 1]))

    groups = [(str(d), [d]) for d in dists] + [("1, 4 and 16 pooled", dists[1:])]
    L += ["\n### D. Consensus: precision of a name by how many of the 4 samples give it\n",
          "| tokens after entity | in 1 of 4 | in 2 of 4 | in 3 of 4 | in 4 of 4 | AUC of the agreement count |", "|---|---|---|---|---|---|"]
    for label, ds in groups:
        cells = []
        for k in (1, 2, 3, 4):
            by = names_at(ds, lambda c: c == k)
            n = sum(map(len, by.values()))
            cells.append(f"{pct(by)} (n={n})" if n else "none")
        L.append(f"| {label} | " + " | ".join(cells) + f" | {auc_ci(ds)} |")

    L += ["\n### E. Filter trade-off: keep only names given by at least k of the 4 samples\n",
          "| tokens after entity | k | wrong names removed | true names kept | precision after filtering | names left |", "|---|---|---|---|---|---|"]
    for label, ds in groups:
        base = names_at(ds)
        L.append(f"| {label} | no filter | 0% | 100% | {pct(base)} | {sum(map(len, base.values()))} |")
        for k in (2, 3, 4):
            removed, kept = defaultdict(list), defaultdict(list)
            for (i, dd), names in per.items():
                if dd in ds:
                    removed[ent[i]] += [c < k for _, c, t in names if not t]
                    kept[ent[i]] += [c >= k for _, c, t in names if t]
            after = names_at(ds, lambda c: c >= k)
            L.append(f"| {label} | {k} | {pct(removed)} | {pct(kept)} | {pct(after)} | {sum(map(len, after.values()))} |")

    L += ["\n### F. Where names sit: describing the context vs predicting the continuation\n",
          "Each explanation is cut at the cue that introduces its predicted continuation (\"strongly expecting ...\"). "
          "Shares are of all explanations at that distance.\n",
          "| tokens after entity | true name, context part | true name, continuation part | wrong name, context part | wrong name, continuation part | wrong names (context / continuation) |",
          "|---|---|---|---|---|---|"]
    for d in dists:
        L.append(f"| {d} | {pct(by_entity(d, lambda r: [r['tc']]))} | {pct(by_entity(d, lambda r: [r['tn']]))} | "
                 f"{pct(by_entity(d, lambda r: [bool(r['wc'])]))} | {pct(by_entity(d, lambda r: [bool(r['wn'])]))} | "
                 f"{sum(len(r['wc']) for r in recs[d])} / {sum(len(r['wn']) for r in recs[d])} |")
    no_cue = sum(not parts(r["text"])[1] for d in dists for r in recs[d])
    L.append(f"\n{no_cue} of {n_exp} explanations have no continuation cue and count as context only.")

    L += ["\n### G. What the wrong names are, at 4 and 16 tokens after the entity\n",
          "| tokens after entity | wrong names in total | one of our listed entities | same-type entity outside our list | most common |",
          "|---|---|---|---|---|"]
    rng = random.Random(0)
    examples = []
    for d in (4, 16):
        names = Counter(n for r in recs[d] for n in r["wrong"])
        listed, tot = sum(c for n, c in names.items() if n in TYPE), sum(names.values())
        L.append(f"| {d} | {tot} | {listed} ({listed / tot:.0%}) | {tot - listed} ({(tot - listed) / tot:.0%}) | "
                 + ", ".join(f"{n} ({c})" for n, c in names.most_common(12)) + " |")
        examples.append((d, rng.sample([r for r in recs[d] if r["wrong"] and not r["true"]], 15)))
    for d, ex in examples:
        L.append(f"\n#### 15 random wrong-name explanations, {d} tokens after the entity\n")
        for j, r in enumerate(ex, 1):
            p = passages[r["i"]]
            L += [f"{j}. **True entity:** {p['entity']} ({p['type']}). **Wrong names given:** {', '.join(sorted(r['wrong']))}",
                  f"   - Passage: {p['text']!r}", f"   - Explanation: {r['text']!r}"]
    (OUT / f"name_analysis{'' if MODE == 'clean' else '-' + MODE}.md").write_text("\n".join(L))
    print("\n".join(l for l in L if not l.startswith("   - ")))
