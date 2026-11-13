"""Fine-tune a LoRA on the released verbalizer so it writes fewer unsupported names.

The target model and the reconstructor stay frozen. One round:
  sample K explanations per activation -> score each -> keep the best one per activation ->
  fine-tune the LoRA on the kept ones (plain LM loss, 1 epoch; --epochs for more) -> the next round samples from the new LoRA.

The score is the experiment; three arms (config files in configs/):
  recon     cosine(h, AR(z))
  holdback  cosine - lam * (names in z that fewer than 2 reference samples also give)
  lineup    AR(z) must pick h out of 4 near-twins; score = log-softmax of cosine / tau over the 4
  recall    holdback + beta if z names the entity that is really in the passage (supervised by the passage
            text, unlike the other arms); --holdout N keeps N entities per type out of training

Defaults that differ from the first written spec (each has a flag to switch back):
  agreement_ref=released  agreement is counted against samples from the frozen released verbalizer, so it
                          cannot rise just because the tuned model's samples become alike ("current" = spec)
  name_types=all          the penalty counts names of all six types, not only the passage's type ("same" = spec)
  negatives=twins         lineup alternatives are the same passage with the entity swapped for another of its
                          type ("other" = other entities' passages, which differ in everything, = spec)

Training data: activations of the 4,200 non-test passages at mixed distances. Test passages are never used here.
Every stage caches under runs/hb/<tag>/ and logs its seconds; rerunning resumes.

    python hb.py configs/holdback.json [--n_train 20 --k 4 --rounds 1 --tag tiny]
"""
import argparse
import json
import multiprocessing as mp
import random
import time
from collections import Counter
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from entities import ENTITIES, TYPE, cfg_hash, mentions
from fade2 import DISTS, _extract_trunc, load_acts, load_passages, split
from names import all_names, claims
from nla_local import AR, AV, DEV, DT, RUNS, TARGET, _verbalize, cos, isolated, load, log
from nla_inference import NLACritic, inject_at_marked_positions, load_nla_config, normalize_activation  # noqa: E402

DEFAULT = dict(arm="recon", n_train=2000, k=8, rounds=3, seed=0, dists=[0, 1, 4, 16], lam=0.01, tau=1.0,
               agreement_ref="released", name_types="all", negatives="twins",
               lr=1e-4, lora_r=16, lora_alpha=32, lora_dropout=0.05, train_batch=8, gen_batch=32, ar_batch=32,
               max_new_tokens=200, epochs=1, beta=0.05, holdout=0, score_workers=7, tag="")
HB = RUNS / "hb"


# ---------------------------------------------------------------- GPU stages (each runs in its own process)
@torch.inference_mode()
def _reconstruct_batch(texts, batch=32):
    """explanations -> [N, d] reconstructor outputs, batched (left-padded so the last position is the read position)."""
    ar = NLACritic(snapshot_download(AR), device=DEV, dtype=DT)
    tok = ar.tokenizer
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([ar.template.format(explanation=texts[i]) for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=True).to(DEV)
        h = ar.backbone.model(**enc, use_cache=False).last_hidden_state[:, -1]
        for j, v in zip(idx, ar.value_head(h).float().cpu()):
            out[j] = v
        if b % (batch * 20) == 0:
            log(f"reconstructed {b}/{len(texts)}")
    alone = ar.reconstruct(texts[order[-1]])
    assert cos(alone, out[order[-1]]) > 0.995, "batched reconstruction differs from unbatched"
    return torch.stack(out)


def _train(vectors, texts, adapter_in, adapter_out, hp):
    """One epoch of LM loss on (activation -> explanation) pairs; saves the LoRA to adapter_out."""
    from peft import LoraConfig, PeftModel, get_peft_model
    path, tok, model = load(AV)
    nla = load_nla_config(path, tok)
    if adapter_in:
        model = PeftModel.from_pretrained(model, adapter_in, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp["lora_dropout"],
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.train()
    emb = model.get_input_embeddings()
    ids = torch.tensor([tok.apply_chat_template(
        [{"role": "user", "content": nla.actor_prompt_template.format(injection_char=nla.injection_char)}],
        tokenize=True, add_generation_prompt=True)])
    with torch.no_grad():
        base = emb(ids.to(DEV)).float().cpu()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    # the released verbalizer's training target format (nla_ref/docs/design.md)
    targets = [tok(f"<explanation>\n{t}\n</explanation>", add_special_tokens=False).input_ids + [end] for t in texts]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"])
    order = []
    for e in range(hp.get("epochs", 1)):  # a fresh shuffle per pass; epoch 0 matches the single-epoch runs
        one = list(range(len(texts)))
        random.Random(hp["seed"] + e).shuffle(one)
        order += one
    losses, t0 = [], time.time()
    for b in range(0, len(order), hp["train_batch"]):
        idx = order[b:b + hp["train_batch"]]
        T = base.shape[1] + max(len(targets[i]) for i in idx)
        x = torch.zeros(len(idx), T, base.shape[2])
        mask = torch.zeros(len(idx), T, dtype=torch.long)
        labels = torch.full((len(idx), T), -100)
        for r, i in enumerate(idx):
            p = inject_at_marked_positions(ids, base, normalize_activation(vectors[i][None].float(), nla.injection_scale),
                                           nla.injection_token_id, nla.injection_left_neighbor_id, nla.injection_right_neighbor_id)[0]
            tgt = torch.tensor(targets[i])
            with torch.no_grad():
                te = emb(tgt.to(DEV)).float().cpu()
            n = base.shape[1] + len(tgt)
            x[r, :n] = torch.cat([p, te])
            mask[r, :n] = 1
            labels[r, base.shape[1]:n] = tgt
        out = model(inputs_embeds=x.to(DEV, DT), attention_mask=mask.to(DEV), labels=labels.to(DEV))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.item())
        if (b // hp["train_batch"]) % 10 == 0:
            log(f"train step {b // hp['train_batch'] + 1}/{-(-len(order) // hp['train_batch'])}: loss {losses[-1]:.4f} "
                f"({time.time() - t0:.0f}s)")
    assert all(l == l for l in losses), "training loss became NaN"
    model.save_pretrained(adapter_out)
    return {"first_loss": losses[0], "last_loss": losses[-1], "mean_loss": sum(losses) / len(losses), "steps": len(losses)}


sample, reconstruct_batch, train, extract = map(isolated, (_verbalize, _reconstruct_batch, _train, _extract_trunc))


# ---------------------------------------------------------------- data
def twin_ids(tok, passage, other, dists):
    """The passage with its entity swapped for `other` -> (token ids, positions for each distance), or None."""
    ms = mentions(passage["text"], passage["entity"])
    if len(ms) != 1:
        return None
    text = passage["text"][:ms[0].start()] + other + passage["text"][ms[0].end():]
    enc = tok(text, return_offsets_mapping=True)
    t = max(i for i, (a, b) in enumerate(enc.offset_mapping) if a < ms[0].start() + len(other))
    return (enc.input_ids, [t + d for d in dists]) if t + max(dists) < len(enc.input_ids) else None


def build_data(cfg):
    """Training activations (non-test passages, mixed distances) and their near-twins. Cached by data config."""
    key = cfg_hash({k: cfg[k] for k in ("n_train", "seed", "dists")})
    f = HB / "shared" / f"data-{key}.pt"
    if f.exists():
        return torch.load(f), key
    passages, _ = load_passages()
    H, _ = load_acts(passages)
    fit, val, _test = split(passages)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot_download(TARGET, allow_patterns=["*.json", "*.txt"]))
    rng = random.Random(cfg["seed"])
    pool = [(i, d) for i in fit + val for d in cfg["dists"]]
    rng.shuffle(pool)
    items, seqs, pos = [], [], []
    for i, d in pool:
        if len(items) == cfg["n_train"]:
            break
        p = passages[i]
        others = rng.sample([e for e in ENTITIES[p["type"]] if e != p["entity"]], 3)
        tw = [twin_ids(tok, p, o, [d]) for o in others]
        if any(t is None for t in tw):
            continue
        items.append(dict(i=i, d=d, entity=p["entity"], type=p["type"], text=p["text"], twins=others))
        seqs += [t[0] for t in tw]
        pos += [t[1] for t in tw]
    assert len(items) == cfg["n_train"], f"only {len(items)} usable training activations"
    t0 = time.time()
    tw = torch.stack([a[0] for a in extract(seqs, pos)]).view(len(items), 3, -1)
    data = dict(items=items, h=torch.stack([H[it["i"], DISTS.index(it["d"])] for it in items]), twins=tw,
                twin_seconds=time.time() - t0)
    # a twin should be close to, but not the same as, its original
    c = cos(data["h"][:, None], tw)
    log(f"twins: mean cosine with the original {c.mean():.3f} (min {c.min():.3f}, max {c.max():.3f})")
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, f)
    return data, key


# ---------------------------------------------------------------- scoring (CPU)
_names_cache = {}


def names_of(text, typ, mode):
    k = (text, typ if mode == "same" else None)
    if k not in _names_cache:
        _names_cache[k] = frozenset((t, n) for t in (ENTITIES if mode == "all" else [typ]) for n in all_names(text, t))
    return _names_cache[k]


def jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / max(1, len(a | b))


def _score_row(job):
    """One activation's K samples -> (unsupported-name count per sample, (names the true entity, wrong names) per sample)."""
    it, row, refrow, same, agreement_ref, name_types = job
    unsup = []
    for s, z in enumerate(row):
        # a sample is never counted as its own support: in round 0 the reference samples are the samples themselves
        others = [r for o, r in enumerate(refrow) if not (same and o == s)] if agreement_ref == "released" else \
                 [r for o, r in enumerate(row) if o != s]
        support = Counter(n for o in others for n in names_of(o, it["type"], name_types))
        unsup.append(sum(support[n] < 2 for n in names_of(z, it["type"], name_types)))
    return unsup, [claims(z, it["text"], it["entity"]) for z in row]


def score_round(cfg, data, samples, recs, ref):
    """-> (scores [N, K], stats). ref[j] = reference samples for activation j (from the released verbalizer)."""
    N, K = len(samples), len(samples[0])
    c = cos(recs.view(N, K, -1), data["h"][:, None])
    jobs = [({k: it[k] for k in ("type", "text", "entity")}, samples[j], ref[j], ref is samples, cfg["agreement_ref"],
             cfg["name_types"]) for j, it in enumerate(data["items"])]
    if cfg["score_workers"] > 1:  # name detection is the slow part and is independent per activation
        with mp.get_context("spawn").Pool(cfg["score_workers"]) as pool:
            rows = pool.map(_score_row, jobs, chunksize=16)
    else:
        rows = list(map(_score_row, jobs))
    unsup, truth = torch.tensor([u for u, _ in rows], dtype=torch.float), [t for _, t in rows]
    if cfg["arm"] == "recon":
        score = c.clone()
    elif cfg["arm"] == "holdback":
        score = c - cfg["lam"] * unsup
    elif cfg["arm"] == "recall":  # supervised: the passage text says which entity is really there
        score = c - cfg["lam"] * unsup + cfg["beta"] * torch.tensor([[float(t) for t, _ in row] for row in truth])
    elif cfg["arm"] == "lineup":
        if cfg["negatives"] == "twins":
            cand = torch.cat([data["h"][:, None], data["twins"]], 1)  # [N, 4, d]
        else:  # three other entities' activations of the same type and distance
            rng, cand = random.Random(cfg["seed"]), []
            for j, it in enumerate(data["items"]):
                pool = [o for o, x in enumerate(data["items"]) if x["type"] == it["type"] and x["d"] == it["d"]
                        and x["entity"] != it["entity"]]
                cand.append(torch.stack([data["h"][j]] + [data["h"][o] for o in rng.sample(pool, 3)]))
            cand = torch.stack(cand)
        cc = cos(recs.view(N, K, 1, -1), cand[:, None])  # [N, K, 4]
        score = torch.log_softmax(cc / cfg["tau"], -1)[..., 0]
    else:
        raise ValueError(cfg["arm"])
    best = score.argmax(1)
    pick = lambda fn: sum(fn(truth[j][best[j]]) for j in range(N)) / N
    allm = lambda fn: sum(fn(t) for row in truth for t in row) / (N * K)
    stats = {
        "cosine_all": c.mean().item(), "cosine_kept": c[range(N), best].mean().item(),
        "score_all": score.mean().item(), "score_kept": score[range(N), best].mean().item(),
        "unsupported_names_all": unsup.mean().item(), "unsupported_names_kept": unsup[range(N), best].mean().item(),
        # the three things that can go wrong: names deleted wholesale, samples collapsing, explanations shrinking
        "true_name_rate_all": allm(lambda t: t[0]), "true_name_rate_kept": pick(lambda t: t[0]),
        "wrong_names_per_explanation_all": allm(lambda t: len(t[1])), "wrong_names_per_explanation_kept": pick(lambda t: len(t[1])),
        "similarity_between_samples": sum(jaccard(row[a], row[b]) for row in samples for a in range(K) for b in range(a))
                                      / (N * K * (K - 1) / 2),
        "identical_sample_pairs": sum(row[a] == row[b] for row in samples for a in range(K) for b in range(a)) / (N * K * (K - 1) / 2),
        "length_chars_all": sum(len(z) for row in samples for z in row) / (N * K),
        "length_chars_kept": sum(len(samples[j][best[j]]) for j in range(N)) / N,
    }
    return score, best, stats


# ---------------------------------------------------------------- the loop
def load_cfg(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    for k, v in DEFAULT.items():
        if k != "dists":
            ap.add_argument(f"--{k}", type=type(v), default=None)
    a = ap.parse_args(argv)
    cfg = {**DEFAULT, **json.loads(Path(a.config).read_text())}
    cfg.update({k: v for k, v in vars(a).items() if k != "config" and v is not None})
    cfg["tag"] = cfg["tag"] or cfg["arm"]
    return cfg


def main(argv=None):
    cfg = load_cfg(argv)
    out = HB / f"{cfg['tag']}-{cfg['arm']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log(f"arm {cfg['arm']}: {cfg['n_train']} activations x K={cfg['k']} x {cfg['rounds']} round(s) -> {out}")
    timings = json.loads((out / "timings.json").read_text()) if (out / "timings.json").exists() else {}

    def timed(name, fn):
        t0 = time.time()
        r = fn()
        timings[name] = round(time.time() - t0, 1)
        (out / "timings.json").write_text(json.dumps(timings, indent=1))
        log(f"stage {name}: {timings[name]}s")
        return r

    data, key = timed("data", lambda: build_data(cfg))
    K, keep = cfg["k"], list(range(cfg["n_train"]))
    if cfg["holdout"]:  # some entities never appear in training, so the eval can tell reading from memorising
        held = sorted(e for t in ENTITIES for e in random.Random(f"{cfg['seed']}-{t}").sample(ENTITIES[t], cfg["holdout"]))
        (out / "holdout.json").write_text(json.dumps(held))
        keep = [j for j, it in enumerate(data["items"]) if it["entity"] not in held]
        data = {"items": [data["items"][j] for j in keep], "h": data["h"][keep], "twins": data["twins"][keep]}
        log(f"held out of training: {', '.join(held)} -> {len(keep)} activations left")
    N = len(keep)
    adapter, ref, history = None, None, []
    for r in range(cfg["rounds"]):
        # round 0 samples the released verbalizer, identically for every arm, so it is shared between arms
        rd = HB / "shared" / f"round0-{key}-k{K}" if r == 0 else out / f"round{r}"
        rd.mkdir(parents=True, exist_ok=True)
        sf, rf = rd / "samples.json", rd / "recs.pt"
        if not sf.exists():
            s = timed(f"round{r}_sample", lambda: sample(data["h"], n=K, batch=cfg["gen_batch"], seed=cfg["seed"] + 1000 * r,
                                                         max_new_tokens=cfg["max_new_tokens"], adapter=adapter,
                                                         live=str(rd / "live.jsonl")))
            sf.write_text(json.dumps(s))
        samples = json.loads(sf.read_text())
        if not rf.exists():
            torch.save(timed(f"round{r}_reconstruct", lambda: reconstruct_batch([z for row in samples for z in row], cfg["ar_batch"])), rf)
        recs = torch.load(rf)
        if r == 0:
            if len(samples) != N:  # the shared round 0 covers every activation; drop the held-out entities' rows
                samples, recs = [samples[j] for j in keep], recs.view(len(samples), K, -1)[keep].reshape(N * K, -1)
            ref = samples
        score, best, stats = timed(f"round{r}_score", lambda: score_round(cfg, data, samples, recs, ref))
        kept = [samples[j][best[j]] for j in range(N)]
        ad = out / f"round{r}" / "adapter"
        if not (ad / "adapter_config.json").exists():
            ad.parent.mkdir(parents=True, exist_ok=True)
            (ad.parent / "kept.json").write_text(json.dumps(kept))
            stats["train"] = timed(f"round{r}_train", lambda: train(data["h"], kept, adapter, str(ad),
                                                                    {k: cfg[k] for k in ("lr", "lora_r", "lora_alpha", "lora_dropout", "train_batch", "seed", "epochs")}))
        adapter = str(ad)
        history.append({"round": r, **stats})
        (out / "history.json").write_text(json.dumps(history, indent=1))
        log(f"round {r} " + ", ".join(f"{k}={v:.3f}" for k, v in stats.items() if isinstance(v, float)))
    print(json.dumps({"timings_seconds": timings, "history": history}, indent=1))


if __name__ == "__main__":
    main()
