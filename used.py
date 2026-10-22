"""Used specifics: prompts whose next word IS the specific, so the model is using it at that token.

The planted-concept runs showed B and F cannot be judged on a concept the model ignores (a poet added
to a bread recipe never reaches the output). Here each prompt ends just before the concept word, e.g.
"...the capital city of France is called" -> Paris. Then, on the real last-token activation h:

  named   does the verbalizer name the upcoming word?
  A       is it top among its same-type candidates in h?
  B       true explanation vs rival-swapped one, by cosine and by KL of the model's next tokens
  F       steer with reader(swapped) - reader(original): does the model now say the rival?

Stages cache under runs/used/.
"""
import json
from collections import defaultdict
from pathlib import Path

import torch

from checks_ar import boot, edit, rivals
from checks_target import K, target_stage, top_in_type
from nla_local import cos, extract, reconstruct, verbalize
from planted import CONCEPTS, N, named

OUT = Path(__file__).parent / "runs" / "used"
ALPHAS = (1.0, 4.0, 8.0)
LEAD = ("Here are some quick facts I wrote down while studying for tomorrow's general knowledge quiz. I always "
        "mix these up under pressure, so I am going to state each one plainly and carefully, one at a time, "
        "and then read the whole list back to myself twice before bed. Fact one: ")
PROMPTS = {
    "Baudelaire": "the French poet who wrote the collection Les Fleurs du mal was Charles",
    "Verlaine": "the French poet who wrote Poèmes saturniens and had a stormy affair with Rimbaud was Paul",
    "Shakespeare": "the English playwright who wrote Hamlet and Macbeth was William",
    "Whitman": "the American poet who wrote Leaves of Grass was Walt",
    "Paris": "the capital city of France is called",
    "Tokyo": "the capital city of Japan is called",
    "Cairo": "the capital city of Egypt is called",
    "Chicago": "the large city in Illinois known as the Windy City is called",
    "rabbit": "the small long-eared animal that hops and is linked with Easter is called a",
    "mouse": "the tiny rodent that cats chase and that loves cheese is called a",
    "elephant": "the huge grey animal with a trunk and tusks is called an",
    "dolphin": "the clever marine mammal that leaps beside boats and uses echolocation is called a",
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names, texts = list(PROMPTS), [LEAD + p for p in PROMPTS.values()]

    f = OUT / "acts.pt"
    if not f.exists():
        torch.save(torch.stack([a[-1] for a in extract(texts)[0]]), f)
    hs = torch.load(f)

    f = OUT / "explanations.json"
    if not f.exists():
        f.write_text(json.dumps(verbalize(hs, n=N)))
    exps = json.loads(f.read_text())

    f = OUT / "recs.pt"
    recs = torch.load(f) if f.exists() else {}
    need = {e for es in exps for e in es}
    need |= {edit(e, c, x) for c, es in zip(names, exps) for e in es if c in named(e) for x in rivals(c)}
    need = sorted(need - recs.keys())
    if need:
        recs.update(zip(need, reconstruct(need)))
        torch.save(recs, f)

    # one job per sample that names the concept: alts = [true rebuild, 3 rival rebuilds, h steered toward one rival per alpha]
    jobs, meta = [], []
    for i, (c, es) in enumerate(zip(names, exps)):
        for k, e in enumerate(es):
            if c not in named(e):
                continue
            h, x = hs[i], rivals(c)[k % 3]
            at = lambda text: recs[text] / recs[text].norm() * h.norm()
            steer = [h + a * (at(edit(e, c, x)) - at(e)) for a in ALPHAS]
            jobs.append({"carrier": i, "ref": h, "gen": True, "gen_tokens": 8, "alts": [at(e)] + [at(edit(e, c, x)) for x in rivals(c)] + steer})
            meta.append((i, c, e, x))
    jobs.append({"carrier": 0, "ref": hs[0], "gen": False, "alts": []})  # keeps the stage valid if nothing was named
    f = OUT / "target.pt"
    if not f.exists():
        torch.save(target_stage(jobs, texts=texts), f)
    res = torch.load(f)

    print("\n== named: does the verbalizer name the word the model is about to say?")
    print(f"per sample:   {boot([c in named(e) for c, es in zip(names, exps) for e in es])}")
    print(f"any of {N}:     {boot([any(c in named(e) for e in es) for c, es in zip(names, exps)])}")
    print(f"same-type rival named instead: {boot([bool(named(e) & set(rivals(c))) for c, es in zip(names, exps) for e in es])}")
    for c, es in zip(names, exps):
        print(f"  {c:<12}{sum(c in named(e) for e in es)}/{N}   others named: {sorted(set().union(*map(named, es)) - {c})}")

    print("\n== A: is the upcoming word top among its 4 same-type candidates in h? (chance 0.25)")
    print(boot([top_in_type(hs[i], res["mu"], res["verify"], c) for i, c in enumerate(names)]))

    print("\n== C: near-twin lineup over the 4 same-type prompts (chance 0.25)")
    line = defaultdict(list)
    for i, (c, es) in enumerate(zip(names, exps)):
        for e in es:
            win = all(cos(recs[e], hs[i]) > cos(recs[e], hs[names.index(x)]) for x in rivals(c))
            line["names it" if c in named(e) else "does not name it"].append(win)
    for k, v in line.items():
        print(f"{k:<18}{boot(v)}")

    print(f"\n== B: true explanation vs rival-swapped one, by cosine and by KL over the next {K} tokens")
    cos_d, kl_d, kl_true = [], [], []
    flips = defaultdict(list)
    model_says = []
    for (i, c, e, sx), out in zip(meta, res["jobs"]):
        kl_true.append(out["kl"][0])
        model_says.append(c in named(out["ref_text"]))
        for j, x in enumerate(rivals(c)):
            cos_d.append(cos(recs[e], hs[i]).item() - cos(recs[edit(e, c, x)], hs[i]).item())
            kl_d.append(out["kl"][1 + j] - out["kl"][0])
        for a, text in zip(ALPHAS, out["texts"][4:]):
            flips[(a, "rival named")].append(sx in named(text))
            flips[(a, "original named")].append(c in named(text))
    std = lambda xs: torch.tensor(xs).std().item()
    print(f"model's own continuation names the word: {boot(model_says)}")
    print(f"true wins by cosine:      {boot([d > 0 for d in cos_d])}")
    print(f"true wins by KL:          {boot([d > 0 for d in kl_d])}")
    if len(cos_d) > 1:
        print(f"cosine margin:            {boot(cos_d)}   standardized {sum(cos_d) / len(cos_d) / std(cos_d):.2f}")
        print(f"KL penalty for the rival: {boot(kl_d)}   standardized {sum(kl_d) / len(kl_d) / std(kl_d):.2f}")
    print(f"KL of the true rebuild itself: {boot(kl_true)}")

    print("\n== F: steer h with reader(swapped) - reader(original); what does the model say next?")
    for k in sorted(flips):
        print(f"alpha={k[0]:<4} {k[1]:<15}{boot(flips[k])}")
