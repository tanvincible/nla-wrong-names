"""H: theme then residual (run planted.py and checks_ar.py first).

For planted activations where none of the 4 first-pass explanations named the concept: subtract the
reader's rebuild of the best first-pass explanation from h and verbalize what is left. Baseline with the
same budget: just sample the whole activation 4 more times.
"""
import json

import torch

from checks_ar import OUT, PLANTED, boot, load
from nla_local import cos, verbalize
from planted import N, named

if __name__ == "__main__":
    rows, vecs = load()
    recs = torch.load(OUT / "recs.pt")
    cvec = torch.load(PLANTED / "acts.pt")["cvec"]
    cases = [r for r in rows if r["cond"] == "planted" and r["strength"] >= 1.0
             and not any(r["concept"] in named(e) for e in r["samples"])]

    resid = []
    for r in cases:
        h = vecs[r["id"]]
        best = max(r["samples"], key=lambda e: cos(recs[e], h).item())
        u = recs[best] / recs[best].norm()
        resid.append(h - (h @ u) * u)  # h minus its projection on the theme rebuild
    resid = torch.stack(resid)
    whole = torch.stack([vecs[r["id"]] for r in cases])
    assert (torch.stack([cos(a, b) for a, b in zip(resid, whole)]) < 0.99).all()

    log = OUT / "h.json"
    if not log.exists():
        out = verbalize(torch.cat([resid, whole]), n=N)
        log.write_text(json.dumps({"residual": out[:len(cases)], "resample": out[len(cases):]}))
    out = json.loads(log.read_text())

    print(f"\n== H: {len(cases)} planted activations (strength >= 1) whose concept no first-pass sample named")
    cv = torch.stack([cvec[r["concept"]] for r in cases])
    print(f"cosine with the planted concept vector: whole h {cos(whole, cv).mean():.3f}, residual {cos(resid, cv).mean():.3f}")
    for name in ("residual", "resample"):
        per = [r["concept"] in named(e) for r, es in zip(cases, out[name]) for e in es]
        anyn = [any(r["concept"] in named(e) for e in es) for r, es in zip(cases, out[name])]
        false = [bool(named(e) - {r["concept"]}) for r, es in zip(cases, out[name]) for e in es]
        print(f"{name:<9} concept named per sample: {boot(per)}")
        print(f"{name:<9} concept named in any of {N}: {boot(anyn)}")
        print(f"{name:<9} a different concept named: {boot(false)}")
    for r, es in list(zip(cases, out["residual"]))[:3]:
        print(f"\n[{r['concept']} @ {r['strength']}] residual sample: {es[0][:400]}")
