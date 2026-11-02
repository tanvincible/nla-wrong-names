"""Steps 2-4 on the larger corpus, with a swept probe. Supersedes fade.py (which stays as the 30-per-entity run).

All results are NATURAL text (real Wikipedia passages); nothing is planted.

  split     per entity: 30% test; of the rest, 20% validation and 80% fit
  step 2    readout A: concept vectors ("Tell me about {x}." minus baseline mean), top-1 of the 10 same-type
  step 3    10-way logistic-regression probe per distance and type; C chosen on validation from CS, then
            refit on fit+validation and scored on test
  step 4    verbalizer explanations (generated elsewhere, read from runs/fade2/nla-*.jsonl): names the true
            entity / a different same-type entity / neither; then a verifier test on every same-type claim,
            scored two ways: readout A's margin and the trained probe's margin

    python fade2.py            analysis (extracts activations on first run)
    python fade2.py --export   also write runs/fade2/step4_input.pt for the generation job

Intervals: bootstrap over entities (10k resamples); Wilson for plain proportions. Caches keyed by config hash.
"""
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from entities import ENTITIES, TYPE, boot_mean, cfg_hash, fmt, named, phrase, wilson
from nla_local import DEV, TARGET, cos, isolated, layer, load, log
from planted import BASELINE

OUT = Path(__file__).parent / "runs" / "fade2"
PASSAGES = Path(__file__).parent / "runs" / "fade" / "passages-bb2077b20f.json"  # 100 per entity where Wikipedia allows
DISTS = [0, 1, 2, 4, 8, 16]
CS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]  # widened: the first range hit both edges
ACT_CFG = dict(passages=PASSAGES.name, dists=DISTS, baseline=BASELINE, prompt="Tell me about {}.")
SPLIT = dict(seed=0, test_frac=0.3, val_frac=0.2)
# step 4: test passages per entity, distances, samples per activation
# The 9 are the first 9 of each entity's test list, whose order comes from the seeded shuffle in split():
# fixed by seed before any step 4 output existed.
NLA_CFG = dict(per_entity=9, dists=[0, 1, 4, 16], n=4, temperature=1.0, act=cfg_hash(ACT_CFG), split=SPLIT)
PREDICTION = (
    "**Prediction, stated before the step 4 results.** If the verbalizer names the true entity at distances 4 "
    "and 16 much more often than the probe's top-1 accuracy there, the fade is a limitation of the probe, not a "
    "property of the activation. If it names the true entity no more often than the probe reads it, and its "
    "same-type confabulations rise as the probe falls, the fade is real and the verbalizer is guessing.")


@torch.inference_mode()
def _extract_trunc(id_lists, positions, batch=8):
    """Like nla_local.extract_at, but keeps only the blocks the activation needs (less memory, faster)."""
    L = layer()
    _, tok, model = load(TARGET)
    base = model.model
    del base.layers[L:]
    base.norm = torch.nn.Identity()  # last_hidden_state is then the raw output of block L-1 = hidden_states[L]
    model.lm_head = None
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()

    def run(chunk):
        T = max(map(len, chunk))
        ids = torch.full((len(chunk), T), tok.pad_token_id)
        mask = torch.zeros((len(chunk), T), dtype=torch.long)
        for j, x in enumerate(chunk):  # right-padded: with a causal mask, padding cannot reach real tokens
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        return base(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).last_hidden_state

    order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))  # similar lengths per batch
    out = [None] * len(id_lists)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        hs = run([id_lists[i] for i in idx])
        for j, i in enumerate(idx):
            out[i] = hs[j, positions[i]].float().cpu()
        if b % (batch * 25) == 0:
            log(f"extract {b}/{len(order)} sequences")
    alone = run([id_lists[order[-1]]])[0, positions[order[-1]]].float().cpu()
    assert cos(alone, out[order[-1]]).min() > 0.999, "batched activations differ from unbatched"
    return out


def load_acts(passages):
    f = OUT / f"acts-{cfg_hash(ACT_CFG)}.pt"
    if not f.exists():
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(snapshot_download(TARGET, allow_patterns=["*.json", "*.txt"]))
        chat = lambda w: tok.apply_chat_template([{"role": "user", "content": ACT_CFG["prompt"].format(w)}],
                                                 tokenize=True, add_generation_prompt=True)
        prompts = [chat(phrase(e)) for e in TYPE] + [chat(w) for w in BASELINE]
        acts = isolated(_extract_trunc)([p["ids"] for p in passages] + prompts,
                                        [[p["t"] + d for d in DISTS] for p in passages] + [[len(x) - 1] for x in prompts])
        last = torch.stack([a[0] for a in acts[len(passages):]])
        torch.save({"H": torch.stack(acts[:len(passages)]).half(),
                    "cv": dict(zip(TYPE, last[:len(TYPE)] - last[len(TYPE):].mean(0)))}, f)
    d = torch.load(f)
    H = d["H"].float()
    # the truncated-model extraction must agree with the full-model one used for the first corpus
    old_p, old_a = OUT.parent / "fade" / "passages-a5b4344c76.json", OUT.parent / "fade" / "acts-d47225e0f3.pt"
    if old_p.exists() and old_a.exists():
        old = {tuple(p["ids"]): i for i, p in enumerate(json.loads(old_p.read_text())["passages"])}
        both = [(i, old[tuple(p["ids"])]) for i, p in enumerate(passages) if tuple(p["ids"]) in old][:200]
        if both:
            Ho = torch.load(old_a)["H"].float()
            c = cos(H[[a for a, _ in both]], Ho[[b for _, b in both]])
            assert c.min() > 0.995, f"truncated extraction disagrees with the earlier one (min cosine {c.min():.4f})"
            print(f"extraction check: {len(both)} passages shared with the first corpus, min cosine {c.min():.4f}")
    return H, d["cv"]


def load_passages():
    """Every entity capped at the same count (the smallest entity's), so the 10 classes per type are balanced."""
    by = defaultdict(list)
    for p in json.loads(PASSAGES.read_text())["passages"]:
        by[p["entity"]].append(p)
    cap = min(map(len, by.values()))
    return [p for e in by for p in by[e][:cap]], cap


def split(passages):
    """per entity -> (fit, val, test) index lists."""
    by = defaultdict(list)
    for i, p in enumerate(passages):
        by[p["entity"]].append(i)
    fit, val, test = [], [], []
    for e, idx in by.items():
        random.Random(f"{SPLIT['seed']}-{e}").shuffle(idx)
        k = round(len(idx) * SPLIT["test_frac"])
        v = round((len(idx) - k) * SPLIT["val_frac"])
        test += idx[:k]
        val += idx[k:k + v]
        fit += idx[k + v:]
    return fit, val, test


def a_scores(h, mu, cv, typ):
    """readout A: cosine of the centred activation with each same-type concept vector."""
    return {e: cos(h - mu, cv[e]).item() for e in ENTITIES[typ]}


def margin(scores, e):
    return scores[e] - max(v for k, v in scores.items() if k != e)


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    OUT.mkdir(parents=True, exist_ok=True)
    passages, cap = load_passages()
    H, cv = load_acts(passages)
    fit, val, test = split(passages)
    train = fit + val
    ent = [p["entity"] for p in passages]
    n_per = defaultdict(int)
    for e in ent:
        n_per[e] += 1
    mu = H[train].mean(0)
    assert set(n_per.values()) == {cap}
    lines = [f"Passages: {len(passages)} ({len(fit)} fit, {len(val)} validation, {len(test)} test), {len(n_per)} entities, "
             f"{cap} per entity (every entity capped at the same count). All natural text.\n"]

    # ---- steps 2 and 3
    A, P = [defaultdict(list) for _ in DISTS], [defaultdict(list) for _ in DISTS]
    probes, chosen_c = {}, defaultdict(list)
    X = H.numpy()
    for k, d in enumerate(DISTS):
        for i in test:
            s = a_scores(H[i, k], mu[k], cv, TYPE[ent[i]])
            A[k][ent[i]].append(max(s, key=s.get) == ent[i])
        for typ in ENTITIES:
            sel = lambda idx: [i for i in idx if TYPE[ent[i]] == typ]
            f_, v_, tr, te = sel(fit), sel(val), sel(train), sel(test)
            sc = StandardScaler().fit(X[f_, k])
            acc = {}
            for C in CS:
                clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[f_, k]), [ent[i] for i in f_])
                acc[C] = clf.score(sc.transform(X[v_, k]), [ent[i] for i in v_])
            best = max(CS, key=lambda C: (acc[C], -C))  # ties go to the stronger regularization
            chosen_c[d].append(best)
            sc = StandardScaler().fit(X[tr, k])
            clf = LogisticRegression(C=best, max_iter=3000).fit(sc.transform(X[tr, k]), [ent[i] for i in tr])
            probes[(d, typ)] = (sc, clf)
            for i, pred in zip(te, clf.predict(sc.transform(X[te, k]))):
                P[k][ent[i]].append(pred == ent[i])
        log(f"probe sweep done for distance {d}: chosen C per type {chosen_c[d]}")
    a_ci, p_ci = [boot_mean(x) for x in A], [boot_mean(x) for x in P]
    lines += ["### Steps 2 and 3: top-1 accuracy among the 10 same-type entities (natural, test passages, chance 0.10)\n",
              f"Probe: logistic regression, C chosen on validation from {CS} per distance and type, refit on fit+validation.\n",
              "| tokens after entity | readout A (concept vectors) | trained probe (swept) | C chosen per type |", "|---|---|---|---|"]
    lines += [f"| {d} | {fmt(a)} | {fmt(p)} | {', '.join(f'{c:g}' for c in chosen_c[d])} |" for d, a, p in zip(DISTS, a_ci, p_ci)]

    # ---- step 4 rows: the first per_entity test passages of each entity, at each distance
    by_e = defaultdict(list)
    for i in test:
        by_e[ent[i]].append(i)
    chosen = [i for e in by_e for i in by_e[e][:NLA_CFG["per_entity"]]]
    rows = [(i, d) for d in NLA_CFG["dists"] for i in chosen]
    tag = cfg_hash(NLA_CFG)
    if "--export" in sys.argv:
        torch.save({"rows": rows, "n": NLA_CFG["n"], "temperature": NLA_CFG["temperature"],
                    "vectors": torch.stack([H[i, DISTS.index(d)] for i, d in rows]).half()}, OUT / f"step4_input-{tag}.pt")
        print(f"wrote step4_input-{tag}.pt: {len(rows)} activations x {NLA_CFG['n']} samples = {len(rows) * NLA_CFG['n']} explanations")
    done = {}
    for f in sorted(OUT.glob(f"nla-{tag}*.jsonl")):
        for j in map(json.loads, f.read_text().splitlines()):
            done[(j["i"], j["d"])] = j["samples"]
    rows = [r for r in rows if r in done]
    lines += ["\n" + PREDICTION]

    ci = None
    if rows:
        from names import classify
        kinds = ("true", "other", "neither")
        # two classifiers: "narrow" sees only our 10 entities per type; "broad" sees any same-type name
        both = {v: {d: {k: defaultdict(list) for k in kinds} for d in NLA_CFG["dists"]} for v in ("narrow", "broad")}
        like = {v: {d: defaultdict(list) for d in NLA_CFG["dists"]} for v in ("narrow", "broad")}
        claims = []  # (entity of passage, distance, claim is true, A margin, probe margin)
        for i, d in rows:
            e, k, typ = ent[i], DISTS.index(d), TYPE[ent[i]]
            s = a_scores(H[i, k], mu[k], cv, typ)
            sc, clf = probes[(d, typ)]
            ps = dict(zip(clf.classes_, clf.decision_function(sc.transform(X[[i], k]))[0]))
            for text in done[(i, d)]:
                for v, kind in zip(("narrow", "broad"), classify(text, passages[i]["text"], e)):
                    for name in kinds:
                        both[v][d][name][e].append(name == kind)
                    if kind != "neither":  # like-for-like with the probe: it named something of the type
                        like[v][d][e].append(kind == "true")
                claims += [(e, d, c == e, margin(s, c), margin(ps, c)) for c in named(text, typ)]
        cls = both["narrow"]
        counts = {d: {k: sum(map(sum, v.values())) for k, v in cls[d].items()} for d in cls if cls[d]["true"]}
        ci = {d: {k: boot_mean(v) for k, v in cls[d].items()} for d in counts}
        lines += [f"\n{len(rows)} activations, {sum(len(done[r]) for r in rows)} explanations."]
        for v, what in (("narrow", "wrong name = one of the 9 other listed entities of the type"),
                        ("broad", "wrong name = any same-type entity (all countries, large cities, any person, WordNet "
                                  "animals/foods/instruments) that is not the true one and does not occur in the passage")):
            lines += [f"\n### Step 4: what the verbalizer names, {v} detection (natural, test passages)\n", f"{what.capitalize()}.\n",
                      "| tokens after entity | names the true entity | names a wrong same-type entity | names neither | counts (true / wrong / neither) |",
                      "|---|---|---|---|---|"]
            for d in counts:
                c = {k: sum(map(sum, x.values())) for k, x in both[v][d].items()}
                r = {k: fmt(boot_mean(x)) for k, x in both[v][d].items()}
                lines.append(f"| {d} | {r['true']} | {r['other']} | {r['neither']} | {c['true']} / {c['other']} / {c['neither']} |")
        lines += ["\n### Step 4: like-for-like with the probe (natural)\n",
                  "The probe is forced to pick one of 10. The comparable verbalizer number is: among explanations that "
                  "name some entity of the right type, the share that name the true one.\n",
                  "| tokens after entity | probe top-1 | verbalizer, narrow (n) | verbalizer, broad (n) |", "|---|---|---|---|"]
        for d in counts:
            cell = lambda v: (f"{fmt(boot_mean(like[v][d]))} ({sum(map(len, like[v][d].values()))})" if like[v][d] else "n/a (0)")
            lines.append(f"| {d} | {fmt(p_ci[DISTS.index(d)])} | {cell('narrow')} | {cell('broad')} |")

        def auc_ci(sub, col):
            by = defaultdict(list)
            for c in sub:
                by[c[0]].append((c[2], c[col]))
            auc = lambda xs: roc_auc_score([t for t, _ in xs], [m for _, m in xs]) if 0 < sum(t for t, _ in xs) < len(xs) else float("nan")
            keys, rng = list(by), random.Random(0)
            ms = sorted(m for m in (auc([v for k in rng.choices(keys, k=len(keys)) for v in by[k]]) for _ in range(10000)) if m == m)
            return (auc([v for k in keys for v in by[k]]), ms[int(.025 * len(ms))], ms[int(.975 * len(ms)) - 1]) if ms else (float("nan"),) * 3

        lines += ["\n### Step 4: can a verifier separate true from false claims? (natural)\n",
                  "A claim is any same-type entity an explanation names. It passes if the activation scores the claimed "
                  "entity above the 9 others. Scored two ways: readout A (concept vectors) and the trained probe.\n",
                  "| tokens after entity | scorer | true claims | pass rate, true (Wilson) | false claims | pass rate, false (Wilson) | AUC (bootstrap over entities) |",
                  "|---|---|---|---|---|---|---|"]
        for d in ["all"] + list(counts):
            sub = [c for c in claims if d == "all" or c[1] == d]
            for name, col in (("readout A", 3), ("trained probe", 4)):
                tp, fp = [c[col] > 0 for c in sub if c[2]], [c[col] > 0 for c in sub if not c[2]]
                lines.append(f"| {d} | {name} | {len(tp)} | {fmt(wilson(sum(tp), len(tp)))} | {len(fp)} | "
                             f"{fmt(wilson(sum(fp), len(fp)))} | {fmt(auc_ci(sub, col))} |")

        rng = random.Random(0)
        lines.append("\n### Step 4: 10 random explanations per distance (natural)\n")
        for d in counts:
            lines.append(f"#### {d} tokens after the entity\n")
            pool = [r for r in rows if r[1] == d]
            for i, _ in rng.sample(pool, min(10, len(pool))):
                p, text = passages[i], rng.choice(done[(i, d)])
                lines += [f"- **Entity:** {p['entity']} ({p['type']}); the explanation names: "
                          f"{', '.join(sorted(named(text, p['type']))) or 'none of the 10'}",
                          f"  - Passage (cut 16 tokens after the entity): {p['text']!r}",
                          f"  - Explanation: {text!r}"]

    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = range(len(DISTS))

    def draw(pos, cis, label, **kw):
        m, lo, hi = zip(*cis)
        ax.plot(pos, m, marker="o", label=label, **kw)
        ax.fill_between(pos, lo, hi, alpha=0.15, color=kw.get("color"))
    draw(xs, a_ci, "readout A (concept vectors), top-1 of 10", color="tab:blue")
    draw(xs, p_ci, "trained probe (swept), top-1 of 10", color="tab:green")
    if ci:
        pos = [DISTS.index(d) for d in ci]
        draw(pos, [ci[d]["true"] for d in ci], "verbalizer names the true entity", color="tab:orange", linestyle="--")
        draw(pos, [ci[d]["other"] for d in ci], "verbalizer names a different same-type entity", color="tab:red", linestyle="--")
    ax.axhline(0.1, color="gray", linewidth=0.8, linestyle=":", label="chance for top-1 of 10")
    ax.set_xticks(list(xs), DISTS)
    ax.set_xlabel("tokens after the entity's last token")
    ax.set_ylabel("rate (95% interval over entities)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Natural text: how fast an entity fades from the activation (output of block 20)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fade.png", dpi=150)
    lines.append("\n![fade plot](runs/fade2/fade.png)\n")
    (OUT / "section.md").write_text("\n".join(lines))
    print("\n".join(l for l in lines if "Passage (cut" not in l and "Explanation:" not in l))
