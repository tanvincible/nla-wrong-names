"""Step 5: do B (KL) and F (causal edit) work when the model is actually using the specific?

All results here are NATURAL prompts (nothing planted). 120 quiz questions (2 per entity), each read at
two positions, plus 12 rhyming couplets:

  group a   read at "...A: The answer to this question is"  -> the entity is the NEXT token
  group b   read at "...A:"                                 -> the entity comes 6 tokens later
  couplet   read at the newline after the first line        -> the rhyme word ends the second line

A prompt is "valid" only if the model's own greedy continuation names the entity where expected
(completed within the first 3 tokens for a; at token index >= 3 for b and couplets). Results use valid
prompts only. Logit lens = final norm + unembedding applied to the layer-20 activation.
Caches are keyed by config hash under runs/used2/; writes runs/used2/section.md.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from checks_target import K, target_stage
from entities import ENTITIES, TYPE, boot_mean, cfg_hash, fmt, named, swap
from nla_local import DEV, TARGET, cos, isolated, layer, load as load_model, reconstruct, verbalize

OUT = Path(__file__).parent / "runs" / "used2"
HEAD = ("Here is a general knowledge quiz. Each answer is given as one short sentence.\n\n"
        "Q: Which planet is known as the Red Planet?\nA: The answer to this question is Mars.\n\n"
        "Q: Which metal is liquid at room temperature?\nA: The answer to this question is mercury.\n\n"
        "Q: Which ocean is the largest on Earth?\nA: The answer to this question is the Pacific.\n\n"
        "Q: {}\nA:")
A_TAIL = " The answer to this question is"
QUESTIONS = {
    "France": ["Which country has Paris as its capital city?", "Which country is home to the Eiffel Tower and the Louvre?"],
    "Japan": ["Which country has Tokyo as its capital city?", "Which island country is home to Mount Fuji?"],
    "Egypt": ["Which country has Cairo as its capital city?", "Which country is home to the Great Pyramid of Giza?"],
    "Brazil": ["Which country has Brasília as its capital city?", "Which is the largest country in South America?"],
    "Canada": ["Which country has Ottawa as its capital city?", "Which country lies directly north of the United States?"],
    "India": ["Which country has New Delhi as its capital city?", "Which country is home to the Taj Mahal?"],
    "Germany": ["Which country has Berlin as its capital city?", "Which country did Angela Merkel lead as chancellor?"],
    "Italy": ["Which country has Rome as its capital city?", "Which country is shaped like a boot and is home to Venice and Florence?"],
    "Mexico": ["Which country lies directly south of the United States?", "Which country is home to the ancient Aztec capital Tenochtitlan?"],
    "Russia": ["Which country has Moscow as its capital city?", "Which is the largest country in the world by land area?"],
    "Paris": ["Which city is the capital of France?", "In which city does the Eiffel Tower stand?"],
    "Tokyo": ["Which city is the capital of Japan?", "Which city hosted the Summer Olympics in 1964 and again in 2021?"],
    "Cairo": ["Which city is the capital of Egypt?", "Which large African city lies on the Nile near the pyramids of Giza?"],
    "Chicago": ["Which city in Illinois is known as the Windy City?", "In which American city does the Willis Tower, once called the Sears Tower, stand?"],
    "London": ["Which city is the capital of the United Kingdom?", "In which city does Big Ben stand?"],
    "Berlin": ["Which city is the capital of Germany?", "Which city was divided by a wall from 1961 to 1989?"],
    "Moscow": ["Which city is the capital of Russia?", "In which city does the Kremlin stand?"],
    "Sydney": ["Which Australian city is famous for its Opera House and Harbour Bridge?", "Which is the largest city in Australia?"],
    "Toronto": ["Which is the largest city in Canada?", "In which Canadian city does the CN Tower stand?"],
    "Rome": ["Which city is the capital of Italy?", "In which city does the Colosseum stand?"],
    "Shakespeare": ["Which playwright wrote Hamlet and Macbeth?", "Which English writer is known as the Bard of Avon?"],
    "Napoleon": ["Which French emperor was defeated at the Battle of Waterloo?", "Which French general crowned himself emperor in 1804?"],
    "Einstein": ["Which physicist developed the theory of general relativity?", "Which scientist is famous for the equation E equals m c squared?"],
    "Mozart": ["Which composer wrote The Marriage of Figaro and Don Giovanni?", "Which Austrian child prodigy composer was born in Salzburg in 1756?"],
    "Darwin": ["Which naturalist wrote On the Origin of Species?", "Which scientist sailed on HMS Beagle and proposed natural selection?"],
    "Lincoln": ["Which American president issued the Emancipation Proclamation?", "Which American president delivered the Gettysburg Address?"],
    "Churchill": ["Which British prime minister led the United Kingdom through most of the Second World War?", "Which British leader gave the 'we shall fight on the beaches' speech?"],
    "Gandhi": ["Which leader of the Indian independence movement was famous for nonviolent resistance?", "Which Indian leader led the Salt March in 1930?"],
    "Picasso": ["Which Spanish artist painted Guernica?", "Which painter co-founded Cubism with Georges Braque?"],
    "Beethoven": ["Which composer wrote the Moonlight Sonata and nine symphonies while going deaf?", "Which German composer wrote the Ode to Joy in his Ninth Symphony?"],
    "dog": ["Which domesticated animal is known as man's best friend?", "Which common pet barks and wags its tail?"],
    "cat": ["Which common pet purrs and meows?", "Which small domesticated feline is kept to catch mice?"],
    "horse": ["Which animal do jockeys ride in races such as the Kentucky Derby?", "Which large animal is ridden with a saddle and kept in a stable?"],
    "lion": ["Which large African predator has a mane and is known as the king of the jungle?", "Which animal lives in a pride and has a male with a large mane?"],
    "tiger": ["Which large striped predator with orange and black fur is native to Asia?", "Which striped animal is the national animal of India?"],
    "elephant": ["Which is the largest living land animal?", "Which animal has a trunk and ivory tusks?"],
    "wolf": ["Which wild canine lives in packs and howls at the moon?", "Which animal dresses up as the grandmother in Little Red Riding Hood?"],
    "eagle": ["Which bird of prey is the national bird of the United States?", "Which large bird of prey with a white head appears on the Great Seal of the United States?"],
    "rabbit": ["Which small long-eared animal hops and is associated with Easter?", "What kind of animal is Peter in the Beatrix Potter tale?"],
    "dolphin": ["Which intelligent marine mammal uses echolocation and leaps beside boats?", "What kind of animal was Flipper in the 1960s television series?"],
    "bread": ["Which staple food is baked from flour, water and yeast?", "Which food is sliced to make toast and sandwiches?"],
    "cheese": ["Which dairy food includes varieties such as cheddar, brie and gouda?", "Which food is made by curdling milk and then aging the curds?"],
    "rice": ["Which grain is the staple food of much of Asia and is grown in flooded paddies?", "Which grain is used to make sushi and risotto?"],
    "pasta": ["Which Italian staple food comes in shapes such as spaghetti, penne and fusilli?", "Which food made from durum wheat dough is boiled and served with sauce?"],
    "chocolate": ["Which sweet food is made from roasted cacao beans?", "Which sweet treat comes in dark, milk and white varieties?"],
    "pizza": ["Which Italian dish is a flat round base topped with tomato sauce and mozzarella and baked in an oven?", "Which food is delivered in a flat square box and cut into triangular slices?"],
    "butter": ["Which dairy product is made by churning cream?", "Which yellow dairy spread is commonly put on toast?"],
    "honey": ["Which sweet food do bees make from nectar?", "Which sweet golden liquid is collected from beehives?"],
    "potato": ["Which vegetable is used to make French fries and chips?", "Which starchy tuber was the crop that failed in the Irish famine of the 1840s?"],
    "sugar": ["Which sweet crystalline substance is extracted from cane or beets?", "Which sweetener is commonly stirred into tea and coffee by the spoonful?"],
    "piano": ["Which keyboard instrument has 88 black and white keys?", "Which instrument did Chopin and Liszt famously compose for and perform on?"],
    "guitar": ["Which six-stringed instrument is strummed or plucked and comes in acoustic and electric forms?", "Which instrument did Jimi Hendrix famously play?"],
    "violin": ["Which is the smallest and highest-pitched instrument of the standard string quartet?", "Which instrument did Stradivari famously make and Paganini famously play?"],
    "cello": ["Which large string instrument is played seated, held between the knees?", "Which instrument is Yo-Yo Ma famous for playing?"],
    "flute": ["Which woodwind instrument is held sideways and played by blowing across a hole?", "Which silver woodwind instrument has no reed and is played by James Galway?"],
    "trumpet": ["Which brass instrument with three valves did Louis Armstrong and Miles Davis play?", "Which brass instrument plays the highest notes in the brass section and has three valves?"],
    "saxophone": ["Which reed instrument made of brass was invented by Adolphe Sax?", "Which instrument did John Coltrane and Charlie Parker play?"],
    "clarinet": ["Which single-reed woodwind instrument did Benny Goodman play?", "Which black woodwind instrument with a single reed opens Gershwin's Rhapsody in Blue?"],
    "harp": ["Which large triangular instrument has 47 strings that are plucked with the fingers?", "Which stringed instrument is the national symbol of Ireland?"],
    "banjo": ["Which stringed instrument with a round drum-like body is central to bluegrass music?", "Which instrument does Kermit the Frog play while singing Rainbow Connection?"],
}
COUPLET_HEAD = "Here is a short poem made of rhyming couplets. Every second line ends with a word that rhymes with the line before it.\n\n"
COUPLETS = {
    "rabbit": "He saw a carrot and had to grab it,", "cat": "On the windowsill beside my hat,",
    "dog": "Something barked out in the fog,", "bread": "The baker rose early and got out of bed,",
    "cheese": "The mouse crept out and whispered please,", "rice": "For dinner tonight something simple and nice,",
    "honey": "The bees work hard but not for money,", "flute": "The musician stood in a velvet suit,",
    "harp": "The angel's fingers were quick and sharp,", "France": "They flew to Europe to learn to dance,",
    "Rome": "After the long trip he felt at home,", "horse": "The rider galloped around the course,",
}
CFG = dict(head=HEAD, a_tail=A_TAIL, questions=QUESTIONS, couplet_head=COUPLET_HEAD, couplets=COUPLETS,
           n=4, temperature=1.0, rivals=3, jobs_per_item=2, alphas=[1.0, 4.0, 8.0], seed=0, k=K)
GEN = {"a": 8, "b": 16, "couplet": 16}


def items():
    out = []
    for e, qs in QUESTIONS.items():
        for q in qs:
            out.append({"entity": e, "group": "b", "text": HEAD.format(q)})
            out.append({"entity": e, "group": "a", "text": HEAD.format(q) + A_TAIL})
    out += [{"entity": e, "group": "couplet", "text": COUPLET_HEAD + line + "\n"} for e, line in COUPLETS.items()]
    return out


@torch.inference_mode()
def _lens(texts, first_ids, k=16):
    """Per text: layer-L activation at the last token, logit-lens and final-layer rank of the entity's first
    token, and the greedy continuation as a list of growing prefixes."""
    _, tok, model = load_model(TARGET)
    L, out = layer(), []
    for text, fid in zip(texts, first_ids):
        ids = tok(text, return_tensors="pt").to(DEV)
        o = model(**ids, output_hidden_states=True)
        h = o.hidden_states[L][0, -1]
        rank = lambda logits: int((logits.float() > logits.float()[fid]).sum())
        gen = model.generate(**ids, max_new_tokens=k, do_sample=False, repetition_penalty=1.0,
                             pad_token_id=tok.eos_token_id)[0, ids.input_ids.shape[1]:]
        out.append({"h": h.float().cpu(), "lens_rank": rank(model.lm_head(model.model.norm(h))),
                    "final_rank": rank(o.logits[0, -1]), "cont": [tok.decode(gen[:j + 1]) for j in range(len(gen))]})
    return out


lens = isolated(_lens)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tag = cfg_hash(CFG)
    its = items()
    rng = random.Random(CFG["seed"])

    f = OUT / f"lens-{tag}.pt"
    if not f.exists():
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(snapshot_download(TARGET, allow_patterns=["*.json", "*.txt"]))
        torch.save(lens([it["text"] for it in its], [tok(" " + it["entity"]).input_ids[0] for it in its]), f)
    for it, r in zip(its, torch.load(f)):
        it.update(r)
        typ = TYPE[it["entity"]]
        at = next((j for j, c in enumerate(it["cont"]) if it["entity"] in named(c, typ)), None)  # token completing it
        it["valid"] = at is not None and (at <= 2 if it["group"] == "a" else at >= 3)
        it["at"] = at

    f = OUT / f"explanations-{tag}.json"
    if not f.exists():
        f.write_text(json.dumps(verbalize(torch.stack([it["h"] for it in its]), n=CFG["n"], temperature=CFG["temperature"])))
    for it, es in zip(its, json.loads(f.read_text())):
        it["samples"] = es

    # B and F jobs: up to jobs_per_item samples that name the entity, on valid prompts
    jobs, meta, need = [], [], set()
    for i, it in enumerate(its):
        e = it["entity"]
        naming = [s for s in it["samples"] if e in named(s, TYPE[e])][:CFG["jobs_per_item"]] if it["valid"] else []
        for s in naming:
            rv = rng.sample([x for x in ENTITIES[TYPE[e]] if x != e], CFG["rivals"])
            meta.append((i, s, rv))
            need |= {s} | {swap(s, e, x) for x in rv}
    f = OUT / f"recs-{tag}.pt"
    recs = torch.load(f) if f.exists() else {}
    need |= {s for it in its for s in it["samples"]}
    todo = sorted(need - recs.keys())
    if todo:
        recs.update(zip(todo, reconstruct(todo)))
        torch.save(recs, f)
    for i, s, rv in meta:
        it = its[i]
        h, e = it["h"], it["entity"]
        at_h = lambda text: recs[text] / recs[text].norm() * h.norm()
        steer = [h + a * (at_h(swap(s, e, rv[0])) - at_h(s)) for a in CFG["alphas"]]
        jobs.append({"carrier": i, "ref": h, "gen": True, "gen_tokens": GEN[it["group"]],
                     "alts": [at_h(s)] + [at_h(swap(s, e, x)) for x in rv] + steer})
    f = OUT / f"target-{tag}.pt"
    if not f.exists():
        torch.save(target_stage(jobs, texts=[it["text"] for it in its])["jobs"] if jobs else [], f)
    res = torch.load(f)

    lines = ["All prompts are natural (nothing planted). Intervals: bootstrap over entities, 10k resamples.\n",
             "| | group a: entity is the next token | group b: entity comes 6 tokens later | couplets: entity ends the next line |",
             "|---|---|---|---|"]
    row = defaultdict(dict)
    for g in ("a", "b", "couplet"):
        sub = [it for it in its if it["group"] == g]
        val = [it for it in sub if it["valid"]]
        by = lambda fn, pool=None: fmt(boot_mean(_group(fn, val if pool is None else pool))) if (val if pool is None else pool) else "n/a"

        def _group(fn, pool):
            d = defaultdict(list)
            for it in pool:
                d[it["entity"]] += fn(it)
            return d
        nm = lambda it: [it["entity"] in named(s, TYPE[it["entity"]]) for s in it["samples"]]
        row["prompts (valid / total)"][g] = f"{len(val)} / {len(sub)}"
        row["logit lens at layer 20: entity is top-1"][g] = by(lambda it: [it["lens_rank"] == 0])
        row["logit lens at layer 20: entity in top-5"][g] = by(lambda it: [it["lens_rank"] < 5])
        row["final layer: entity is the top next token"][g] = by(lambda it: [it["final_rank"] == 0])
        row["verbalizer names the entity (per sample)"][g] = by(nm)
        row["verbalizer names the entity (any of 4)"][g] = by(lambda it: [any(nm(it))])
        row["verbalizer names a different same-type entity"][g] = by(
            lambda it: [bool(named(s, TYPE[it["entity"]]) - {it["entity"]}) for s in it["samples"]])
        row["verbalizer names it, when logit lens top-5 has it"][g] = by(nm, [it for it in val if it["lens_rank"] < 5])
        row["verbalizer names it, when logit lens top-5 lacks it"][g] = by(nm, [it for it in val if it["lens_rank"] >= 5])

        cw, kw, cm, km, klt = (defaultdict(list) for _ in range(5))
        flips = defaultdict(lambda: defaultdict(list))
        for (i, s, rv), out in zip(meta, res):
            it = its[i]
            if it["group"] != g:
                continue
            e, h, typ = it["entity"], it["h"], TYPE[it["entity"]]
            klt[e].append(out["kl"][0])
            for j, x in enumerate(rv):
                dc = cos(recs[s], h).item() - cos(recs[swap(s, e, x)], h).item()
                dk = out["kl"][1 + j] - out["kl"][0]
                cw[e].append(dc > 0); kw[e].append(dk > 0); cm[e].append(dc); km[e].append(dk)
            flips[("original named", 0.0)][e].append(e in named(out["ref_text"], typ))
            flips[("rival named", 0.0)][e].append(rv[0] in named(out["ref_text"], typ))
            for a, text in zip(CFG["alphas"], out["texts"][1 + len(rv):]):
                flips[("original named", a)][e].append(e in named(text, typ))
                flips[("rival named", a)][e].append(rv[0] in named(text, typ))
        b = lambda d: fmt(boot_mean(d)) if d else "n/a"
        row["B/F: explanations tested"][g] = str(sum(1 for (i, _, _) in meta if its[i]["group"] == g))
        row["B: true explanation beats rival-swapped, by cosine"][g] = b(cw)
        row[f"B: true beats rival-swapped, by KL over next {K} tokens"][g] = b(kw)
        row["B: cosine margin (true minus rival)"][g] = b(cm)
        row["B: KL penalty for the rival (nats)"][g] = b(km)
        row["B: KL of the true rebuild itself (nats)"][g] = b(klt)
        for a in [0.0] + CFG["alphas"]:
            label = "no steering" if a == 0 else f"steering x{a:g}"
            row[f"F, {label}: model names the rival"][g] = b(flips[("rival named", a)])
            row[f"F, {label}: model still names the original"][g] = b(flips[("original named", a)])
    lines += [f"| {k} | {v.get('a', '')} | {v.get('b', '')} | {v.get('couplet', '')} |" for k, v in row.items()]
    (OUT / "section.md").write_text("\n".join(lines))
    print("\n".join(lines))
