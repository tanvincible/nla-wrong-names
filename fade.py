"""Steps 2-4: how fast does a named entity fade from the activation, and does NLA confabulation track it?

All results here are NATURAL text (real Wikipedia passages); nothing is planted.

  step 2  readout A: concept vectors ("Tell me about {x}." minus baseline mean), true entity vs the 9
          others of its type, top-1 per distance after the entity's last token (chance 10%)
  step 3  a 10-way logistic-regression probe per distance and type, trained on the train passages
  step 4  the released verbalizer on test passages: names the true entity / a different same-type
          entity / neither; then A's rival test on every same-type claim

Passages are split 70/30 within each entity; everything is reported on test only. Intervals are
bootstraps over entities (10k resamples), Wilson for plain proportions. Caches are keyed by config hash.
Writes runs/fade/section.md and runs/fade/fade.png.
"""
import json
import random
import sys
from collections import defaultdict

import numpy as np
import torch

from build_passages import OUT
from entities import ENTITIES, TYPE, boot_entities, boot_mean, cfg_hash, fmt, named, phrase, wilson
from nla_local import TARGET, cos, extract_at, verbalize
from planted import BASELINE

DISTS = [0, 1, 2, 4, 8, 16]
PATH = OUT / "passages-a5b4344c76.json"  # the corpus this analysis is pinned to
PILOT = "--pilot" in sys.argv  # 40 test passages, 1 sample: proves the step 4 pipeline before a long run
ACT_CFG = dict(passages=PATH.name, dists=DISTS, baseline=BASELINE, prompt="Tell me about {}.")
SPLIT = dict(seed=0, test_frac=0.3)
# step 4 size: test passages per entity (of 9), distances, samples per activation, generation batch
NLA_CFG = dict(per_entity=9, dists=[0, 1, 4, 16], n=4, temperature=1.0, batch=8, act=cfg_hash(ACT_CFG), split=SPLIT)
if PILOT:
    NLA_CFG = dict(NLA_CFG, pilot_passages=40, n=1)


def load_acts(passages):
    f = OUT / f"acts-{cfg_hash(ACT_CFG)}.pt"
    if not f.exists():
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(snapshot_download(TARGET, allow_patterns=["*.json", "*.txt"]))
        chat = lambda w: tok.apply_chat_template([{"role": "user", "content": ACT_CFG["prompt"].format(w)}],
                                                 tokenize=True, add_generation_prompt=True)
        prompts = [chat(phrase(e)) for e in TYPE] + [chat(w) for w in BASELINE]
        acts = extract_at([p["ids"] for p in passages] + prompts,
                          [[p["t"] + d for d in DISTS] for p in passages] + [[len(x) - 1] for x in prompts])
        last = torch.stack([a[0] for a in acts[len(passages):]])
        torch.save({"H": torch.stack(acts[:len(passages)]).half(),
                    "cv": dict(zip(TYPE, last[:len(TYPE)] - last[len(TYPE):].mean(0)))}, f)
    d = torch.load(f)
    return d["H"].float(), d["cv"]


def split(passages):
    """70/30 within each entity -> (train indices, test indices)."""
    by = defaultdict(list)
    for i, p in enumerate(passages):
        by[p["entity"]].append(i)
    train, test = [], []
    for e, idx in by.items():
        random.Random(f"{SPLIT['seed']}-{e}").shuffle(idx)
        k = round(len(idx) * SPLIT["test_frac"])
        test += idx[:k]
        train += idx[k:]
    return train, test


def a_scores(h, mu, cv, typ):
    """readout A: cosine of the centred activation with each same-type concept vector."""
    return {e: cos(h - mu, cv[e]).item() for e in ENTITIES[typ]}


def a_margin(scores, e):
    return scores[e] - max(v for k, v in scores.items() if k != e)


if __name__ == "__main__":
    passages = json.loads(PATH.read_text())["passages"]
    H, cv = load_acts(passages)
    train, test = split(passages)
    ent = [p["entity"] for p in passages]
    mu = H[train].mean(0)  # [dists, d] generic baseline from train passages only
    lines = [f"Passages: {len(passages)} ({len(train)} train, {len(test)} test), {len(set(ent))} entities. All natural text.\n"]

    # ---- steps 2 and 3
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    A, P = [defaultdict(list) for _ in DISTS], [defaultdict(list) for _ in DISTS]
    for k in range(len(DISTS)):
        for i in test:
            s = a_scores(H[i, k], mu[k], cv, TYPE[ent[i]])
            A[k][ent[i]].append(max(s, key=s.get) == ent[i])
        for typ in ENTITIES:
            tr = [i for i in train if TYPE[ent[i]] == typ]
            te = [i for i in test if TYPE[ent[i]] == typ]
            sc = StandardScaler().fit(H[tr, k].numpy())
            clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(H[tr, k].numpy()), [ent[i] for i in tr])
            for i, pred in zip(te, clf.predict(sc.transform(H[te, k].numpy()))):
                P[k][ent[i]].append(pred == ent[i])
    a_ci, p_ci = [boot_mean(x) for x in A], [boot_mean(x) for x in P]
    lines += ["### Steps 2 and 3: top-1 accuracy among the 10 same-type entities (natural, test passages, chance 0.10)\n",
              "| tokens after entity | readout A (concept vectors) | trained probe |", "|---|---|---|"]
    lines += [f"| {d} | {fmt(a)} | {fmt(p)} |" for d, a, p in zip(DISTS, a_ci, p_ci)]

    # ---- step 4
    by_e = defaultdict(list)
    for i in test:
        by_e[ent[i]].append(i)
    chosen = [i for e in by_e for i in by_e[e][:NLA_CFG["per_entity"]]]
    if PILOT:
        chosen = random.Random(0).sample(sorted(test), NLA_CFG["pilot_passages"])
    rows = [(i, d) for d in NLA_CFG["dists"] for i in chosen]
    log = OUT / f"nla-{cfg_hash(NLA_CFG)}.jsonl"
    done = {(j["i"], j["d"]): j["samples"] for j in map(json.loads, log.read_text().splitlines())} if log.exists() else {}
    todo = [r for r in rows if r not in done]
    print(f"step 4: {len(chosen)} passages x {len(NLA_CFG['dists'])} distances x {NLA_CFG['n']} samples = "
          f"{len(rows) * NLA_CFG['n']} explanations, {len(todo) * NLA_CFG['n']} still to generate", flush=True)
    if "--no-generate" in sys.argv:
        todo = []
    chunk = 256 // NLA_CFG["n"]  # activations per model load; results saved after each
    for c in range(0, len(todo), chunk):
        part = todo[c:c + chunk]
        out = verbalize(torch.stack([H[i, DISTS.index(d)] for i, d in part]), n=NLA_CFG["n"],
                        temperature=NLA_CFG["temperature"], batch=NLA_CFG["batch"])
        with log.open("a") as fh:
            for (i, d), s in zip(part, out):
                done[(i, d)] = s
                fh.write(json.dumps({"i": i, "d": d, "samples": s}) + "\n")
    rows = [r for r in rows if r in done]

    if rows:
        cls = {d: {k: defaultdict(list) for k in ("true", "other", "neither")} for d in NLA_CFG["dists"]}
        claims = []  # (entity of passage, distance, is the claim true, A margin for the claimed entity)
        for i, d in rows:
            e, k = ent[i], DISTS.index(d)
            s = a_scores(H[i, k], mu[k], cv, TYPE[e])
            for text in done[(i, d)]:
                nm = named(text, TYPE[e])
                kind = "true" if e in nm else "other" if nm else "neither"
                for name in cls[d]:
                    cls[d][name][e].append(name == kind)
                claims += [(e, d, c == e, a_margin(s, c)) for c in nm]
        counts = {d: {k: sum(map(sum, v.values())) for k, v in cls[d].items()} for d in cls}
        lines += ["\n### Step 4: raw classification counts (natural)\n", "| tokens after entity | true | other same-type | neither |",
                  "|---|---|---|---|"] + [f"| {d} | {c['true']} | {c['other']} | {c['neither']} |" for d, c in counts.items()]
        lines += ["\n### Step 4: what the verbalizer names (natural, test passages, %d sample(s) per activation)\n" % NLA_CFG["n"],
                  f"{len(rows)} activations, {len(rows) * NLA_CFG['n']} explanations.\n",
                  "| tokens after entity | names the true entity | names a different same-type entity | names neither |",
                  "|---|---|---|---|"]
        ci = {d: {k: boot_mean(v) for k, v in cls[d].items()} for d in cls}
        lines += [f"| {d} | {fmt(ci[d]['true'])} | {fmt(ci[d]['other'])} | {fmt(ci[d]['neither'])} |" for d in cls]

        from sklearn.metrics import roc_auc_score

        def auc(items):
            y = [t for t, _ in items]
            return roc_auc_score(y, [m for _, m in items]) if 0 < sum(y) < len(y) else float("nan")

        def auc_ci(sub):
            by = defaultdict(list)
            for e, _, t, m in sub:
                by[e].append((t, m))
            stat = lambda xs: auc(xs)
            keys = list(by)
            rng = random.Random(0)
            ms = [stat([v for k in rng.choices(keys, k=len(keys)) for v in by[k]]) for _ in range(10000)]
            ms = sorted(m for m in ms if m == m)
            return (auc([v for k in keys for v in by[k]]), ms[int(.025 * len(ms))], ms[int(.975 * len(ms)) - 1]) if ms else (float("nan"),) * 3

        lines += ["\n### Step 4: does A's rival test separate true from false claims? (natural)\n",
                  "A claim is any same-type entity an explanation names. It passes if the activation points at the "
                  "claimed entity more than at the 9 others.\n",
                  "| tokens after entity | true claims | pass rate, true (Wilson) | false claims | pass rate, false (Wilson) | AUC (bootstrap over entities) |",
                  "|---|---|---|---|---|---|"]
        for d in ["all"] + NLA_CFG["dists"]:
            sub = [c for c in claims if d == "all" or c[1] == d]
            tp, fp = [c[3] > 0 for c in sub if c[2]], [c[3] > 0 for c in sub if not c[2]]
            lines.append(f"| {d} | {len(tp)} | {fmt(wilson(sum(tp), len(tp)))} | {len(fp)} | "
                         f"{fmt(wilson(sum(fp), len(fp)))} | {fmt(auc_ci(sub))} |")

        rng = random.Random(0)
        lines.append("\n### Step 4: 10 random explanations per distance (natural)\n")
        for d in NLA_CFG["dists"]:
            lines.append(f"#### {d} tokens after the entity\n")
            for i, _ in rng.sample([r for r in rows if r[1] == d], min(10, len(chosen))):
                p = passages[i]
                text = rng.choice(done[(i, d)])
                nm = sorted(named(text, p["type"]))
                lines += [f"- **Entity:** {p['entity']} ({p['type']}); the explanation names: {', '.join(nm) or 'none of the 10'}",
                          f"  - Passage (cut 16 tokens after the entity): {p['text']!r}",
                          f"  - Explanation: {text!r}"]

    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = range(len(DISTS))

    def draw(pos, ci, label, **kw):
        m, lo, hi = zip(*ci)
        ax.plot(pos, m, marker="o", label=label, **kw)
        ax.fill_between(pos, lo, hi, alpha=0.15, color=kw.get("color"))
    draw(xs, a_ci, "readout A (concept vectors), top-1 of 10", color="tab:blue")
    draw(xs, p_ci, "trained probe, top-1 of 10", color="tab:green")
    if rows:
        pos = [DISTS.index(d) for d in NLA_CFG["dists"]]
        draw(pos, [ci[d]["true"] for d in NLA_CFG["dists"]], "verbalizer names the true entity", color="tab:orange", linestyle="--")
        draw(pos, [ci[d]["other"] for d in NLA_CFG["dists"]], "verbalizer names a different same-type entity", color="tab:red", linestyle="--")
    ax.axhline(0.1, color="gray", linewidth=0.8, linestyle=":", label="chance for top-1 of 10")
    ax.set_xticks(list(xs), DISTS)
    ax.set_xlabel("tokens after the entity's last token")
    ax.set_ylabel("rate (95% interval over entities)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Natural text: how fast an entity fades from the layer-20 activation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / ("pilot-fade.png" if PILOT else "fade.png"), dpi=150)
    lines.append("\n![fade plot](runs/fade/fade.png)\n")
    (OUT / ("pilot-section.md" if PILOT else "section.md")).write_text("\n".join(lines))
    print("\n".join(l for l in lines if PILOT or ("Passage (cut" not in l and "Explanation:" not in l)))
