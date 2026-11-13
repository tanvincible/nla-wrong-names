"""Run the released Qwen2.5-7B NLA on a Mac (MPS), one model in memory at a time.

    extract(texts)        target model  -> layer-20 activations
    verbalize(vectors)    AV            -> explanations
    reconstruct(texts)    AR            -> predicted activations

Each call runs in its own process: three ~15 GB models don't fit in 24 GB together, and MPS only
returns a model's memory when the process exits. Scripts using this need an `if __name__ == "__main__":` guard.
`python nla_local.py` runs a round-trip smoke test.
"""
import json
import multiprocessing as mp
import os
import sys
import time

# cap MPS at macOS's recommended limit so an oversized run errors out instead of swapping the machine to a crawl
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")  # must be <= high, default is 1.4
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "nla_ref"))
from nla_inference import (  # noqa: E402
    EXPLANATION_RE, NLACritic, inject_at_marked_positions, load_nla_config, normalize_activation,
)

# a stage can return thousands of tensors; the default Linux strategy opens one file descriptor each
torch.multiprocessing.set_sharing_strategy("file_system")
DEV = "cuda" if torch.cuda.is_available() else "mps"
# bf16 by default: Qwen's rare ~14k-norm activations can overflow fp16. A T4 has no bf16, so the Kaggle job
# sets NLA_DTYPE=float16 and NLA_DEVICE_MAP=balanced (one 15 GB model split across its two GPUs).
DT = getattr(torch, os.environ.get("NLA_DTYPE", "bfloat16"))
DEVICE_MAP = os.environ.get("NLA_DEVICE_MAP") or {"": DEV}
TARGET = "Qwen/Qwen2.5-7B-Instruct"
AV = "kitft/nla-qwen2.5-7b-L20-av"
AR = "kitft/nla-qwen2.5-7b-L20-ar"


RUNS = Path(__file__).parent / "runs"


def log(msg):
    """Timestamped progress line to stdout and runs/progress.log (watch with: tail -f runs/progress.log)."""
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    RUNS.mkdir(exist_ok=True)
    with (RUNS / "progress.log").open("a") as f:
        f.write(line + "\n")


def load(repo):
    log(f"loading {repo} on {DEV}")
    path = snapshot_download(repo)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=DT, device_map=DEVICE_MAP).eval()  # straight to MPS, no CPU copy
    return path, tok, model


def isolated(fn):
    def run(*args, **kwargs):
        with mp.get_context("spawn").Pool(1) as pool:
            return pool.apply(fn, args, kwargs)
    return run


def layer():
    """Index into HF hidden_states of the activation the NLA was trained on. The sidecar's
    extraction_layer_index K means the OUTPUT of block K (nla_ref/nla/datagen/extractors.py), which is
    hidden_states[K + 1]; the README's hidden_states[K] example is one layer too early."""
    return yaml.safe_load((Path(snapshot_download(AR)) / "nla_meta.yaml").read_text())["extraction_layer_index"] + 1


@torch.inference_mode()
def _extract(texts, chat=False):
    """-> (list of [T, d] fp32 residual-stream activations, list of token-string lists).
    chat=True wraps each text as a user turn and ends at the assistant generation prompt."""
    L = layer()
    _, tok, model = load(TARGET)
    acts, toks = [], []
    for t in texts:
        if chat:
            t = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
        ids = tok(t, return_tensors="pt").to(DEV)
        acts.append(model(**ids, output_hidden_states=True).hidden_states[L][0].float().cpu())
        toks.append(tok.convert_ids_to_tokens(ids["input_ids"][0]))
    return acts, toks


@torch.inference_mode()
def _extract_at(id_lists, positions, batch=8):
    """Token-id lists and, per list, the token indices to read -> list of [len(indices), d] fp32 activations."""
    L = layer()
    _, tok, model = load(TARGET)

    def run(chunk):
        T = max(map(len, chunk))
        ids = torch.full((len(chunk), T), tok.pad_token_id)
        mask = torch.zeros((len(chunk), T), dtype=torch.long)
        for j, x in enumerate(chunk):  # right-padded: with a causal mask, padding cannot reach real tokens
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        # model.model: skip the unembedding, whose logits alone are >1 GB per batch
        return model.model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV), output_hidden_states=True).hidden_states[L]

    out = []
    for i in range(0, len(id_lists), batch):
        hs = run(id_lists[i:i + batch])
        out += [hs[j, pos].float().cpu() for j, pos in enumerate(positions[i:i + batch])]
        if i % (batch * 5) == 0:
            log(f"extract {i}/{len(id_lists)} sequences")
    alone = run(id_lists[:1])[0, positions[0]].float().cpu()
    assert cos(alone, out[0]).min() > 0.999, "batched activations differ from unbatched"
    return out


@torch.inference_mode()
def _verbalize(vectors, n=1, temperature=1.0, max_new_tokens=200, batch=int(os.environ.get("NLA_BATCH", 8)),
               seed=0, live=None, adapter=None):
    """vectors [N, d] -> N lists of n sampled explanations. Batch b is seeded with seed + b. If `live` is a
    path, every batch is appended to it as {"offset", "explanations"} so a crash loses nothing.
    `adapter` is a LoRA checkpoint directory to apply on top of the released verbalizer."""
    path, tok, model = load(AV)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
        log(f"applied adapter {adapter}")
    cfg = load_nla_config(path, tok)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": cfg.actor_prompt_template.format(injection_char=cfg.injection_char)}],
        tokenize=True, add_generation_prompt=True,
    )
    ids = torch.tensor([ids])
    base = model.get_input_embeddings()(ids.to(DEV)).float().cpu()
    out = []
    for chunk in vectors.float().repeat_interleave(n, 0).split(batch):
        emb = torch.cat([
            inject_at_marked_positions(
                ids, base, normalize_activation(v[None], cfg.injection_scale),
                cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id)
            for v in chunk
        ]).to(DEV, DT)
        # plain temperature sampling, as in the reference recipe (overrides Qwen's top_p/top_k defaults)
        t0 = time.time()
        torch.manual_seed(seed + len(out))
        gen = model.generate(
            inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=DEV, dtype=torch.long),
            do_sample=True, temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0,
            max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        for t in tok.batch_decode(gen, skip_special_tokens=True):
            m = EXPLANATION_RE.search(t)
            out.append(m.group(1) if m else t)
        log(f"verbalized {len(out)}/{len(vectors) * n}: batch of {len(chunk)} took {time.time() - t0:.0f}s "
            f"({(time.time() - t0) / len(chunk):.1f}s per explanation) | latest: {out[-1][:120]!r}")
        with open(live or RUNS / "verbalize-live.jsonl", "a") as f:  # every explanation as it is produced
            f.write(json.dumps({"time": time.strftime("%H:%M:%S"), "offset": len(out) - len(chunk),
                                "explanations": out[-len(chunk):]}) + "\n")
    return [out[i:i + n] for i in range(0, len(out), n)]


def _reconstruct(texts):
    """explanations -> [N, d] raw predicted activations."""
    ar = NLACritic(snapshot_download(AR), device=DEV, dtype=DT)
    out = []
    for i, t in enumerate(texts):
        out.append(ar.reconstruct(t))
        if i % 50 == 0:
            log(f"reconstructed {i + 1}/{len(texts)} texts")
    return torch.stack(out)


extract, extract_at, verbalize, reconstruct = map(isolated, (_extract, _extract_at, _verbalize, _reconstruct))


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


if __name__ == "__main__":
    text = ("Charles Baudelaire published Les Fleurs du mal in 1857, and the book was promptly prosecuted "
            "for offending public morals. Six of its poems were banned in France for nearly a century. "
            "Today the collection is regarded as a founding work of modern poetry, and its author as the "
            "first great poet of the modern city, who walked the streets of Paris looking for")
    (acts,), (toks,) = extract([text])
    pos = [len(toks) // 2, len(toks) - 1]
    h = acts[pos]
    exps = verbalize(h, n=2)
    rec = reconstruct([e for es in exps for e in es]).view(len(pos), 2, -1)
    for i, p in enumerate(pos):
        print(f"\n=== token {p} {toks[p]!r}  |h|={h[i].norm():.0f}")
        for j, e in enumerate(exps[i]):
            # other = same explanation scored against the other position's activation (should be lower)
            print(f"--- cos={cos(rec[i, j], h[i]):.3f} (vs other position: {cos(rec[i, j], h[1 - i]):.3f})\n{e}")
