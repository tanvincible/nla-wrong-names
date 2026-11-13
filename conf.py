"""Is agreement between samples just the verbalizer's own confidence? (natural text, no new generation)

Sampling four explanations and counting how many give a name costs four generations. If the count only
estimates the probability the verbalizer assigns to that name, one forward pass should do as well: every
time it wrote a name, it gave that name's tokens a probability.

    python conf.py --score    GPU: run the 8,640 step 4 explanations back through the verbalizer and save
                              the log-probability it assigned to each of their tokens (runs/fade2/conf-logp.pt)
    python conf.py            CPU: score each name by the probability of its FIRST mention in ONE explanation
                              and compare with the agreement count; writes runs/fade2/conf.md

A name's confidence is the joint probability of the tokens of its first mention (later mentions mostly copy
the first). True names are shorter than wrong ones (1.1 tokens against 1.7), so a raw joint probability
separates them even when the log-probs are random numbers (AUC 0.64 in a dry run with stand-in scores). The
AUC reported for it therefore compares only names with the same number of tokens, which a random score puts
at 0.50; the first-token probability, which has no such bias, is reported beside it. All of this was fixed
with stand-in scores, before the real log-probs were looked at. Intervals: bootstrap over entities (10k).
"""
import json
import random
import sys
from collections import defaultdict

import torch

from entities import boot_mean, cfg_hash, fmt
from fade2 import DISTS, NLA_CFG, OUT, load_acts, load_passages
from names import name_spans
from nla_local import AV, DEV, DT, isolated, load, log

LOGP = OUT / "conf-logp.pt"
WRAP = "<explanation>\n"  # the verbalizer's own output format: <explanation>\n{text}\n</explanation><|im_end|>


@torch.inference_mode()
def _token_logprobs(vectors, texts, batch=32):
    """-> per text: (log-prob of each target token given the activation and the tokens before it, char offsets)."""
    from nla_inference import inject_at_marked_positions, load_nla_config, normalize_activation
    path, tok, model = load(AV)
    nla = load_nla_config(path, tok)
    model.eval()
    emb = model.get_input_embeddings()
    ids = torch.tensor([tok.apply_chat_template(
        [{"role": "user", "content": nla.actor_prompt_template.format(injection_char=nla.injection_char)}],
        tokenize=True, add_generation_prompt=True)])
    base = emb(ids.to(DEV)).float().cpu()
    P, end, out = base.shape[1], tok.convert_tokens_to_ids("<|im_end|>"), []
    for b in range(0, len(texts), batch):
        enc = [tok(f"{WRAP}{t}\n</explanation>", add_special_tokens=False, return_offsets_mapping=True) for t in texts[b:b + batch]]
        tgt = [e.input_ids + [end] for e in enc]
        T = P + max(map(len, tgt))
        x = torch.zeros(len(tgt), T, base.shape[2])
        mask = torch.zeros(len(tgt), T, dtype=torch.long)
        for r, t in enumerate(tgt):
            p = inject_at_marked_positions(ids, base, normalize_activation(vectors[b + r][None].float(), nla.injection_scale),
                                           nla.injection_token_id, nla.injection_left_neighbor_id, nla.injection_right_neighbor_id)[0]
            x[r, :P + len(t)] = torch.cat([p, emb(torch.tensor(t).to(DEV)).float().cpu()])
            mask[r, :P + len(t)] = 1
        logits = model(inputs_embeds=x.to(DEV, DT), attention_mask=mask.to(DEV)).logits
        for r, t in enumerate(tgt):
            lp = torch.log_softmax(logits[r, P - 1:P - 1 + len(t)].float(), -1)  # position P-1+j predicts target j
            out.append((lp[range(len(t)), t].cpu(), enc[r].offset_mapping))
        if (b // batch) % 20 == 0:
            log(f"token log-probs {b + len(tgt)}/{len(texts)}")
    return out


token_logprobs = isolated(_token_logprobs)


def load_explanations():
    done = {}
    for f in sorted(OUT.glob(f"nla-{cfg_hash(NLA_CFG)}*.jsonl")):
        for j in map(json.loads, f.read_text().splitlines()):
            done[(j["i"], j["d"])] = j["samples"]
    return done


def first_mentions(text, passage, logp, offsets):
    """{name: (is true, joint log-prob of its first mention, log-prob of its first token, tokens in the mention)}"""
    out = {}
    for a, b, kind, name in name_spans(text, passage["text"], passage["entity"], with_name=True):
        if name in out:
            continue
        a, b = a + len(WRAP), b + len(WRAP)
        toks = [j for j, (s, e) in enumerate(offsets) if s < b and e > a]
        if toks:
            out[name] = (kind == "true", sum(logp[j].item() for j in toks), logp[toks[0]].item(), len(toks))
    return out


def auc(items):
    """Probability that a true name outscores a wrong one (ties count half). items = [(is_true, score)] or
    [(is_true, score, stratum)]; with a stratum, only pairs from the same stratum are compared."""
    import bisect
    strata = defaultdict(lambda: ([], []))
    for y, s, *g in items:
        strata[g[0] if g else 0][0 if y else 1].append(s)
    num = den = 0
    for pos, neg in strata.values():
        neg.sort()
        num += sum(bisect.bisect_left(neg, s) + 0.5 * (bisect.bisect_right(neg, s) - bisect.bisect_left(neg, s)) for s in pos)
        den += len(pos) * len(neg)
    return num / den if den else float("nan")


def auc_ci(by):
    keys, rng = list(by), random.Random(0)
    ms = sorted(m for m in (auc([v for k in rng.choices(keys, k=len(keys)) for v in by[k]]) for _ in range(10000)) if m == m)
    return fmt((auc([v for k in keys for v in by[k]]), ms[int(.025 * len(ms))], ms[int(.975 * len(ms)) - 1]))


def analyse(passages, done, logp):
    keys = sorted(done)
    units = defaultdict(list)  # distance -> [(entity, is_true, joint, first_token, agreement count)]
    for i, d in keys:
        p = passages[i]
        per = [first_mentions(z, p, *logp[(i, d, k)]) for k, z in enumerate(done[(i, d)])]
        for names in per:
            for n, (t, joint, first, ntok) in names.items():
                units[d].append((p["entity"], t, joint, first, sum(n in o for o in per), ntok))
    dists = NLA_CFG["dists"]
    groups = [(str(d), [d]) for d in dists] + [("1, 4 and 16 pooled", dists[1:])]
    L = ["Natural text. Each unit is one name in one explanation (its first mention). 540 passages, 8,640 explanations.\n",
         "### AUC for telling true names from wrong ones\n",
         "| tokens after entity | names (true / wrong) | probability of the name, one explanation (equal-length names compared) | "
         "first token only, one explanation | agreement count over 4 explanations |", "|---|---|---|---|---|"]
    for label, ds in groups:
        u = [x for d in ds for x in units[d]]
        by = lambda col, strat=False: {e: [(x[1], x[col]) + ((x[5],) if strat else ()) for x in u if x[0] == e] for e in {x[0] for x in u}}
        L.append(f"| {label} | {sum(x[1] for x in u)} / {sum(not x[1] for x in u)} | {auc_ci(by(2, True))} | {auc_ci(by(3))} | {auc_ci(by(4))} |")

    L += ["\n### One explanation against four: wrong names removed at the same share of true names kept\n",
          "The confidence threshold (on the raw joint probability, which also benefits from true names being shorter) is set "
          "so that it keeps the same share of true names as the agreement filter does.\n",
          "| tokens after entity | filter | true names kept | wrong names removed, agreement (4 explanations) | "
          "wrong names removed, confidence (1 explanation) |", "|---|---|---|---|---|"]
    for label, ds in groups[1:]:
        u = [x for d in ds for x in units[d]]
        true, wrong = [x for x in u if x[1]], [x for x in u if not x[1]]
        for m in (2, 3, 4):
            kept = sum(x[4] >= m for x in true) / len(true)
            thr = sorted((x[2] for x in true), reverse=True)[max(0, round(kept * len(true)) - 1)]
            group = lambda fn: {e: [fn(x) for x in wrong if x[0] == e] for e in {x[0] for x in wrong}}
            L.append(f"| {label} | at least {m} of 4 | {kept:.1%} | {fmt(boot_mean(group(lambda x: x[4] < m)))} | "
                     f"{fmt(boot_mean(group(lambda x: x[2] < thr)))} |")
    return L


if __name__ == "__main__":
    passages, _ = load_passages()
    done = load_explanations()
    keys = sorted(done)
    if "--score" in sys.argv:
        H, _ = load_acts(passages)
        rows = [(i, d, k) for i, d in keys for k in range(len(done[(i, d)]))]
        res = token_logprobs(torch.stack([H[i, DISTS.index(d)] for i, d, _ in rows]), [done[(i, d)][k] for i, d, k in rows])
        torch.save({(i, d, k): (lp, [tuple(o) for o in off]) for (i, d, k), (lp, off) in zip(rows, res)}, LOGP)
        log(f"saved token log-probs for {len(rows)} explanations to {LOGP}")
        sys.exit()

    L = analyse(passages, done, torch.load(LOGP))
    (OUT / "conf.md").write_text("\n".join(L))
    print("\n".join(L))
