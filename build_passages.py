"""Collect real Wikipedia passages for the fade curve (CPU and disk only; safe beside a GPU job).

Per entity: PER_ENTITY passages where the entity's canonical form appears exactly once, no other listed
entity of the same type appears, with >= BEFORE tokens before the entity's last token and AFTER after it.
Each passage is cut AFTER tokens past the entity. Also collects NEUTRAL passages naming no listed entity.

Source: parquet shards of the wikimedia/wikipedia 20231101.en dump, downloaded to disk and read a few
hundred articles at a time (streaming the dump through `datasets` held 2+ GB of RAM the GPU job needs).

Limits worth knowing: "other entity of the same type" means the other 9 on our list, not every country or
animal in the world; articles whose title contains the entity are skipped; at most one passage per
entity per article; articles are taken in dump order within the chosen shards.
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from entities import ENTITIES, TYPE, cfg_hash, forms, mentions

# NLA_PER_ENTITY=100 builds the larger corpus (reads up to all 41 shards); the default stays the original
# 30-per-entity corpus so its file name, which other scripts are pinned to, does not change.
PER_ENTITY = int(os.environ.get("NLA_PER_ENTITY", 30))
FIRST = [5, 17, 29, 38, 11, 23]
CFG = dict(per_entity=PER_ENTITY, before=50, after=16, max_before=250, neutral=30, neutral_tokens=80,
           source="wikimedia/wikipedia 20231101.en",
           shards=FIRST if PER_ENTITY == 30 else FIRST + [i for i in range(41) if i not in FIRST], entities=ENTITIES)
OUT = Path(__file__).parent / "runs" / "passages"
PATH = OUT / f"passages-{cfg_hash(CFG)}.json"
ANY = re.compile(r"\b(" + "|".join(f for e in TYPE for f in forms(e)) + r")\b", re.I)


def find(tok, title, text, e):
    """First paragraph of the article that qualifies as a passage for entity e, or None."""
    if e.lower() in title.lower():
        return None
    for para in text.split("\n"):
        ms = mentions(para, e)
        if len(para) < 350 or len(ms) != 1 or ms[0].group(1) != e:
            continue
        if any(mentions(para, o) for o in ENTITIES[TYPE[e]] if o != e):
            continue
        enc = tok(para, return_offsets_mapping=True)
        t = max(i for i, (a, b) in enumerate(enc.offset_mapping) if a < ms[0].end())  # entity's last token
        if not (CFG["before"] <= t <= CFG["max_before"]) or len(enc.input_ids) - 1 - t < CFG["after"]:
            continue
        ids = enc.input_ids[:t + CFG["after"] + 1]
        return {"entity": e, "type": TYPE[e], "title": title, "ids": ids, "t": t, "text": tok.decode(ids)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen2.5-7B-Instruct", allow_patterns=["*.json", "*.txt"]))
    got, neutral, n = defaultdict(list), [], 0
    full = lambda: all(len(got[e]) >= CFG["per_entity"] for e in TYPE) and len(neutral) >= CFG["neutral"]
    for shard in CFG["shards"]:
        if full():
            break
        f = hf_hub_download("wikimedia/wikipedia", f"20231101.en/train-{shard:05d}-of-00041.parquet", repo_type="dataset")
        for batch in pq.ParquetFile(f).iter_batches(batch_size=200, columns=["title", "text"]):
            if full():
                break
            for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
                n += 1
                if not ANY.search(text):
                    continue
                for e in TYPE:
                    if len(got[e]) < CFG["per_entity"] and (p := find(tok, title, text, e)):
                        got[e].append(p)
                if len(neutral) < CFG["neutral"] and n % 50 == 0:  # spread neutral passages across articles
                    for para in text.split("\n"):
                        if len(para) >= 350 and not ANY.search(para):
                            ids = tok(para).input_ids
                            if len(ids) >= CFG["neutral_tokens"]:
                                ids = ids[:CFG["neutral_tokens"]]
                                neutral.append({"title": title, "ids": ids, "text": tok.decode(ids)})
                                break
        short = [f"{e} {len(got[e])}" for e in TYPE if len(got[e]) < CFG["per_entity"]]
        print(f"after shard {shard}: {n} articles, {sum(map(len, got.values()))} passages, short: {short}", flush=True)
        PATH.write_text(json.dumps({"cfg": CFG, "passages": [p for e in TYPE for p in got[e]], "neutral": neutral}))
    PATH.write_text(json.dumps({"cfg": CFG, "passages": [p for e in TYPE for p in got[e]], "neutral": neutral}))
    print(f"wrote {PATH}: {sum(map(len, got.values()))} passages, {len(neutral)} neutral")
