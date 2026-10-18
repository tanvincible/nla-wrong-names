"""Checks that need the target model (run planted.py and checks_ar.py first).

  A  verify a named specific against the activation: does h point at the claimed concept more than at
     same-type rivals? Verification vectors come from a different prompt than the planting vectors.
  B  score a rebuild by the model's next-token predictions (KL) instead of cosine: does swapping the true
     name for a rival hurt more clearly under KL than under cosine?
  F  causal edit: steer with reader(edited explanation) - reader(explanation); does the output move, and
     more for true claims than for false ones?

All target-model work happens in one process (target_stage) and is cached in runs/checks/target.pt.
"""
import random
from collections import defaultdict

import torch

from checks_ar import OUT, boot, claims, edit, load, rivals
from nla_local import DEV, TARGET, cos, isolated, layer, load as load_model, log, reconstruct
from planted import BASELINE, CARRIERS, CONCEPTS, named

K = 12  # tokens of continuation compared
ALPHAS = (1.0, 3.0)  # steering strengths for F, in units of the edit vector at |h| scale
NATURAL = [
    CARRIERS[0] + " Anyway, that reminds me: I have been reading a great deal about {} lately, and",
    CARRIERS[1] + " On a different note, my sister came back from a long trip having learned a lot about {}, which",
]
VERIFY_PROMPT = "Write a short paragraph about {}."


@torch.inference_mode()
def _target_stage(jobs, texts=CARRIERS):
    """jobs: [{carrier, ref, alts: [vec], gen}] -> per job {kl: [..], texts: [..]}; plus Idea A activations."""
    _, tok, model = load_model(TARGET)
    L = layer()
    state = {"vec": None, "pos": 0}

    def hook(_, __, out):
        hs = out[0] if isinstance(out, tuple) else out
        if state["vec"] is not None and hs.shape[1] > state["pos"]:  # skipped on cached generation steps
            hs[:, state["pos"]] = state["vec"].to(hs)
        return out
    model.model.layers[L - 1].register_forward_hook(hook)  # its output is hidden_states[L]

    def run(ids, vec, pos):
        state.update(vec=vec, pos=pos)
        return torch.log_softmax(model(ids).logits[0].float(), -1)

    def acts(text, chat=False):
        state["vec"] = None
        if chat:
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(DEV)
        return model(**ids, output_hidden_states=True).hidden_states[L][0].float().cpu()

    cids = [tok(c, return_tensors="pt").input_ids.to(DEV) for c in texts]
    h0 = acts(texts[0])[-1]
    drift = (run(cids[0], h0, cids[0].shape[1] - 1) - run(cids[0], None, 0)).abs().max()
    assert drift < 0.05, f"patching the model's own activation changed its output by {drift}"

    out = []
    for n, job in enumerate(jobs):
        ids, pos = cids[job["carrier"]], cids[job["carrier"]].shape[1] - 1

        def gen(vec, k):
            state.update(vec=vec, pos=pos)
            return model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=k, do_sample=False, repetition_penalty=1.0,
                                  pad_token_id=tok.eos_token_id)
        full = gen(job["ref"], K)
        ref = run(full, job["ref"], pos)[pos:-1]  # predictions for the K continuation tokens
        res = {"kl": [], "texts": [], "ref_text": tok.decode(gen(job["ref"], job.get("gen_tokens", 30))[0, pos + 1:])}
        for v in job["alts"]:
            alt = run(full, v, pos)[pos:-1]
            res["kl"].append((ref.exp() * (ref - alt)).sum(-1).mean().item())
            if job["gen"]:
                res["texts"].append(tok.decode(gen(v, job.get("gen_tokens", 30))[0, pos + 1:]))
        out.append(res)
        if n % 10 == 0:
            log(f"patched-model job {n}/{len(jobs)} | reference continuation: {res['ref_text'][:80]!r}")

    # Idea A: natural mentions (activations from the concept's last token to the end) and verification vectors
    nat = {}
    for t, tmpl in enumerate(NATURAL):
        for c, (_, phrase, _) in CONCEPTS.items():
            prefix = tmpl.split("{}")[0] + phrase
            nat[(t, c)] = acts(tmpl.format(phrase))[len(tok(prefix).input_ids) - 1:]
    last = lambda words: torch.stack([acts(VERIFY_PROMPT.format(w), chat=True)[-1] for w in words])
    vv = last([CONCEPTS[c][1] for c in CONCEPTS]) - last(BASELINE).mean(0)
    mu = torch.cat([acts(c)[10:] for c in CARRIERS]).mean(0)
    return {"jobs": out, "nat": nat, "verify": dict(zip(CONCEPTS, vv)), "mu": mu}


target_stage = isolated(_target_stage)


def top_in_type(h, mu, vecs, c):
    """Idea A's rule: h (centred) points at c more than at any same-type rival."""
    s = {x: cos(h - mu, vecs[x]).item() for x in [c] + rivals(c)}
    return max(s, key=s.get) == c


if __name__ == "__main__":
    rows, vecs = load()
    recs = torch.load(OUT / "recs.pt")
    rng = random.Random(0)
    planted = [(r, e, c, true) for r, e, c, true in claims(rows) if r["cond"] == "planted"]
    true_claims = [x for x in planted if x[3]]
    false_claims = [x for x in planted if not x[3]]
    rng.shuffle(true_claims)
    b_claims, f_claims = true_claims[:60], true_claims[:40] + false_claims
    f_rival = {id(x): rng.choice(rivals(x[2])) for x in f_claims}

    need = sorted({edit(e, c, f_rival[id(x)]) for x in f_claims for _, e, c, _ in [x]} - recs.keys())
    if need:
        recs.update(zip(need, reconstruct(need)))
        torch.save(recs, OUT / "recs.pt")

    jobs, meta = [], []
    for r, e, c, _ in b_claims:
        ref = vecs[r["id"]]
        at = lambda text: recs[text] / recs[text].norm() * ref.norm()
        jobs.append({"carrier": r["carrier"], "ref": ref, "gen": False,
                     "alts": [at(e)] + [at(edit(e, c, x)) for x in rivals(c)]})
        meta.append(("B", r, e, c, None))
    for x in f_claims:
        r, e, c, true = x
        ref = vecs[r["id"]]
        at = lambda text: recs[text] / recs[text].norm() * ref.norm()
        delta = at(edit(e, c, f_rival[id(x)])) - at(e)
        jobs.append({"carrier": r["carrier"], "ref": ref, "gen": True, "alts": [ref + a * delta for a in ALPHAS]})
        meta.append(("F", r, e, c, (true, f_rival[id(x)])))

    cache = OUT / "target.pt"
    if not cache.exists():
        torch.save(target_stage(jobs), cache)
    res = torch.load(cache)
    assert len(res["jobs"]) == len(jobs)

    print("\n== A: on natural text, is the mentioned concept top among its 4 same-type candidates? (chance 0.25)")
    by_off = defaultdict(list)
    for (t, c), a in res["nat"].items():
        for name, i in [("concept token", 0), ("1 token later", 1), ("2 tokens later", 2), ("last token", len(a) - 1)]:
            by_off[name].append(top_in_type(a[i], res["mu"], res["verify"], c))
    for name, v in by_off.items():
        print(f"{name:<16}{boot(v)}")
    print("\n== A: as a filter on the verbalizer's claims: share of claims that pass the check")
    passed = defaultdict(list)
    for r, e, c, true in claims(rows):
        passed[(r["cond"], "true claim" if true else "false claim")].append(
            top_in_type(vecs[r["id"]], res["mu"], res["verify"], c))
    for k in sorted(passed):
        print(f"{k[0]:<9}{k[1]:<13}{boot(passed[k])}")

    print(f"\n== B: true explanation vs rival-swapped one, judged by cosine and by KL over the next {K} tokens")
    cos_d, kl_d = [], []
    for (kind, r, e, c, _), out in zip(meta, res["jobs"]):
        if kind != "B":
            continue
        h = vecs[r["id"]]
        for x, kl_rival in zip(rivals(c), out["kl"][1:]):
            cos_d.append(cos(recs[e], h).item() - cos(recs[edit(e, c, x)], h).item())
            kl_d.append(kl_rival - out["kl"][0])
    std = lambda xs: torch.tensor(xs).std().item()
    print(f"true wins by cosine:      {boot([d > 0 for d in cos_d])}")
    print(f"true wins by KL:          {boot([d > 0 for d in kl_d])}")
    print(f"cosine margin:            {boot(cos_d)}   standardized {sum(cos_d) / len(cos_d) / std(cos_d):.2f}")
    print(f"KL penalty for the rival: {boot(kl_d)}   standardized {sum(kl_d) / len(kl_d) / std(kl_d):.2f}")
    kl_true = [out["kl"][0] for (kind, *_), out in zip(meta, res["jobs"]) if kind == "B"]
    print(f"KL of the true rebuild itself: {boot(kl_true)}")

    print("\n== F: steer with reader(edited) - reader(original); output movement (KL) and what gets named")
    move, names = defaultdict(list), defaultdict(list)
    for (kind, r, e, c, extra), out in zip(meta, res["jobs"]):
        if kind != "F":
            continue
        true, x = extra
        for a, kl, text in zip(ALPHAS, out["kl"], out["texts"]):
            move[("true claim" if true else "false claim", a)].append(kl)
            if true:
                names[(a, "rival named after steering")].append(x in named(text))
                names[(a, "original still named")].append(c in named(text))
        if true:
            names[(0.0, "rival named after steering")].append(x in named(out["ref_text"]))
            names[(0.0, "original still named")].append(c in named(out["ref_text"]))
    for k in sorted(move):
        print(f"KL moved, edit of a {k[0]:<12} alpha={k[1]}: {boot(move[k])}")
    for k in sorted(names):
        print(f"true-claim edits, alpha={k[0]}: {k[1]:<28}{boot(names[k])}")
