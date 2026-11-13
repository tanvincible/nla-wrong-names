"""Check that this pipeline reproduces known reconstruction quality, measured the way an earlier reference run measured it.

Kaggle reference: nla_injection_pilot.py, condition C0_none (2x T4, fp16): mean cosine 0.907,
mean fve_nrm 0.747, mean |h| 111.3. Its recipe, repeated here:
  passages  20 neutral paragraphs written by the target model: "Write a plain paragraph of about 100 words
            about {topic}. No lists, no title." over the pilot's 24 topics; chat template, temperature 0.8,
            top_p 1, top_k 0, 160 new tokens, whitespace collapsed; kept if >= 60 tokens, not a duplicate,
            and free of every synonym of the pilot's 10 concepts; up to 4 rounds, seed 0 + 1000*round + batch
  h         output of block 20 (hidden_states[21]) at the last token
  explain   8 samples per h: rescaled to injection_scale, temperature 1, top_p 1, top_k 0, 200 new tokens,
            stop at eos or <|im_end|>; text = the stripped <explanation> group, else the whole output
  score     NLACritic.score(text, h) -> (mse, cos); fve_nrm = 1 - mse / 0.7335; means over all explanations
The 30 Wikipedia neutral passages from build_passages.py get the same 8-sample scoring as a second row.
Still different from Kaggle: bf16 here (fp16 there), and the sampled text itself.
"""
import json
import os
import re
import sys
import time

import torch
from huggingface_hub import snapshot_download

from build_passages import OUT, PATH
from entities import cfg_hash
from nla_local import AR, AV, DEV, DT, TARGET, extract_at, isolated, load, log, verbalize
# nla_local puts nla_ref on sys.path, so this import must come after it
from nla_inference import (  # noqa: E402
    EXPLANATION_RE, NLACritic, inject_at_marked_positions, load_nla_config, normalize_activation,
)

KAGGLE = dict(cos=0.907, fve=0.747, norm=111.3)
FVE_VAR = 0.7335  # Var(v_nrm) of the Qwen L20 NLA training set, as in the released code
CFG = dict(recipe="kaggle-pilot-C0", n_neutral=20, samples=8, passage_tokens=160, max_new_tokens=200, seed=0,
           wiki=PATH.name, device=DEV, dtype=str(DT))  # device in the key: the pod reruns it

# The pilot's 24 neutral topics and the synonyms of its 10 concepts (no neutral passage may contain one).
NEUTRAL_TOPICS = [
    "rearranging furniture in a small apartment", "a delayed morning commute",
    "organizing a cluttered desk", "learning to use a new spreadsheet program",
    "a quiet afternoon at a public library", "repainting a bedroom wall",
    "planning a monthly household budget", "a power outage in an office building",
    "waiting in line at a government office", "setting up a new laptop",
    "a long meeting about project deadlines", "doing laundry on a busy weekend",
    "fixing a squeaky door hinge", "moving boxes into a storage unit",
    "a software update that took too long", "filing paperwork for a tax return",
    "cleaning windows on a sunny day", "a group project that missed its deadline",
    "sorting old photographs into albums", "a broken elevator in an apartment block",
    "writing a cover letter for a job", "choosing paint colors for a hallway",
    "an afternoon spent answering emails", "replacing the batteries in a smoke detector",
]
SYNONYMS = ["France", "French", "Japan", "Japanese", "dog", "dogs", "puppy", "puppies", "canine", "canines",
            "elephant", "elephants", "pizza", "pizzas", "chocolate", "chocolates", "basketball", "basketballs",
            "tennis", "guitar", "guitars", "guitarist", "guitarists", "piano", "pianos", "pianist", "pianists"]
CONCEPT_RE = re.compile(r"\b(?:" + "|".join(sorted(SYNONYMS, key=len, reverse=True)) + r")\b", re.I)
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


@torch.inference_mode()
def _write_neutral(n, max_new_tokens, seed, rounds=4, batch=32):
    """The pilot's collect_passages for neutral text -> list of {text, ids}."""
    _, tok, model = load(TARGET)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": f"Write a plain paragraph of about 100 words about {t}. No lists, no title."}],
        tokenize=False, add_generation_prompt=True) for t in NEUTRAL_TOPICS]
    kept = []
    for r in range(rounds):
        for i in range(0, len(prompts), batch):
            torch.manual_seed(seed + 1000 * r + i)
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            # temperature/top_p/top_k as the pilot set them; other settings (repetition_penalty) from Qwen's config
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8,
                                 top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
            for g in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                g = " ".join(g.split())
                if len(tok(g)["input_ids"]) >= 60 and not CONCEPT_RE.search(g) and g not in kept:
                    kept.append(g)
        log(f"neutral passages: {len(kept)} kept after round {r + 1}")
        if len(kept) >= n:
            break
    assert len(kept) >= n, f"only {len(kept)} neutral passages passed the filter"
    return [{"text": g, "ids": tok(g)["input_ids"]} for g in kept[:n]]


@torch.inference_mode()
def _explain(vectors, n, seed, max_new_tokens, batch=int(os.environ.get("NLA_BATCH", 8))):
    """The pilot's phase_explain: vectors [N, d] -> N lists of n {text, parsed, cjk}."""
    path, tok, model = load(AV)
    cfg = load_nla_config(path, tok)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": cfg.actor_prompt_template.format(injection_char=cfg.injection_char)}],
        tokenize=True, add_generation_prompt=True,
    )
    ids = torch.tensor([ids])
    base = model.get_input_embeddings()(ids.to(DEV)).float().cpu()
    stop = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if isinstance(i, int) and i >= 0]
    jobs, out = vectors.float().repeat_interleave(n, 0), []
    for b in range(0, len(jobs), batch):
        chunk = jobs[b:b + batch]
        emb = torch.cat([
            inject_at_marked_positions(
                ids, base, normalize_activation(v[None], cfg.injection_scale),
                cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id)
            for v in chunk
        ]).to(DEV, DT)
        torch.manual_seed(seed + b)
        t0 = time.time()
        gen = model.generate(
            inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=DEV, dtype=torch.long),
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
            max_new_tokens=max_new_tokens, eos_token_id=stop, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for raw in tok.batch_decode(gen, skip_special_tokens=True):
            m = EXPLANATION_RE.search(raw)
            text = m.group(1).strip() if m else raw.strip()
            out.append({"text": text, "parsed": bool(m), "cjk": bool(CJK_RE.search(text))})
        log(f"explained {len(out)}/{len(jobs)}: batch of {len(chunk)} took {time.time() - t0:.0f}s "
            f"| latest: {out[-1]['text'][:120]!r}")
    return [out[i:i + n] for i in range(0, len(out), n)]


def _score(texts, vectors):
    """The pilot's phase_score: NLACritic.score(text, h) for each pair -> [N, 2] of (mse, cos)."""
    ar = NLACritic(snapshot_download(AR), device=DEV, dtype=DT)
    out = []
    for i, (t, v) in enumerate(zip(texts, vectors)):
        out.append(ar.score(t, v.numpy()))
        if i % 50 == 0:
            log(f"scored {i + 1}/{len(texts)}")
    return torch.tensor(out)


write_neutral, explain, score = map(isolated, (_write_neutral, _explain, _score))


def run():
    """Both passage sets, one model load per stage; cached by config."""
    f = OUT / f"verify-{cfg_hash(CFG)}.pt"
    if f.exists():
        return torch.load(f)
    wiki = json.loads(PATH.read_text())["neutral"]
    assert len(wiki) >= 20 and all(len(p["ids"]) >= 60 for p in wiki)
    kag = write_neutral(CFG["n_neutral"], CFG["passage_tokens"], CFG["seed"])
    sets = {"kaggle recipe": kag, "wikipedia": wiki}
    ids = [p["ids"] for ps in sets.values() for p in ps]
    h = torch.stack([a[0] for a in extract_at(ids, [[len(x) - 1] for x in ids])])
    exps = [e for es in explain(h, CFG["samples"], CFG["seed"], CFG["max_new_tokens"]) for e in es]
    sc = score([e["text"] for e in exps], h.repeat_interleave(CFG["samples"], 0))
    d, i = {}, 0
    for name, ps in sets.items():
        k = len(ps)
        d[name] = {"texts": [p.get("text") for p in ps], "positions": [len(p["ids"]) - 1 for p in ps],
                   "h": h[i:i + k], "exps": exps[i * CFG["samples"]:(i + k) * CFG["samples"]],
                   "mse": sc[i * CFG["samples"]:(i + k) * CFG["samples"], 0],
                   "cos": sc[i * CFG["samples"]:(i + k) * CFG["samples"], 1]}
        i += k
    torch.save(d, f)
    return d


if __name__ == "__main__":
    d = run()
    S = CFG["samples"]
    rows = {}
    for name, r in d.items():
        cos, fve, norm = r["cos"], 1 - r["mse"] / FVE_VAR, r["h"].norm(dim=-1)
        per = cos.view(-1, S).mean(1)  # passage means: the standard error is over passages
        rows[name] = dict(cos=cos.mean().item(), se=(per.std() / len(per) ** 0.5).item(),
                          fve=fve.mean().item(), norm=norm.mean().item())
        print(f"\n== {name}: {len(per)} passages x {S} samples, read positions "
              f"{min(r['positions'])}-{max(r['positions'])}")
        print(f"mean cosine  {cos.mean():.3f} ± {rows[name]['se']:.3f} (SE over passages; "
              f"min passage {per.min():.3f}, max {per.max():.3f})   Kaggle {KAGGLE['cos']}")
        print(f"mean fve_nrm {fve.mean():.3f}   Kaggle {KAGGLE['fve']}")
        print(f"mean |h|     {norm.mean():.1f}   Kaggle {KAGGLE['norm']}")
        print(f"parse failures {1 - sum(e['parsed'] for e in r['exps']) / len(r['exps']):.1%}, "
              f"CJK {sum(e['cjk'] for e in r['exps']) / len(r['exps']):.1%}   Kaggle (all conditions) 0.0%, 3.9%")

    k = rows["kaggle recipe"]
    ok = k["cos"] >= 0.89 and abs(k["norm"] / KAGGLE["norm"] - 1) <= 0.03
    print(f"\n== verdict on the Kaggle recipe: {'PASS' if ok else 'FAIL'} "
          f"(needs cosine >= 0.89 and |h| within 3% of {KAGGLE['norm']})")
    if not ok:
        raise SystemExit("pipeline check failed: stop and debug before anything else")

    if "--throughput" not in sys.argv:
        raise SystemExit
    h = d["wikipedia"]["h"]
    print(f"\n== throughput at larger batches ({2 * len(h)} explanations each)")
    for b in (16, 32, 64):
        t = time.time()
        try:
            verbalize(h, n=2, batch=b)
            print(f"batch {b}: {(time.time() - t) / 60:.1f} min for {2 * len(h)} explanations including model load",
                  flush=True)
        except Exception as e:  # e.g. the MPS memory cap
            print(f"batch {b}: failed ({str(e)[:200]})", flush=True)
