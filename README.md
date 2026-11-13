# Wrong Names in Natural Language Autoencoder Explanations

Course project for CS787 (Generative Artificial Intelligence), IIT Kanpur, by Tanvi Pooranmal Meena.

**Start with the report: [`paper/main.pdf`](paper/main.pdf).** This repository is the code and the saved
results behind it.

## The project in one minute

A natural language autoencoder (NLA) explains an activation inside a language model by writing a paragraph
about it. A second model must rebuild the activation from that paragraph, which is how the explainer is
trained. The paragraphs read well, but they sometimes name a specific that was never in the text: "piano"
when the passage said "cello".

Using the released NLA for Qwen2.5-7B and 6,000 Wikipedia passages, the project asks how often that
happens, why, and what removes it.

| | Finding |
|---|---|
| How often | Sixteen tokens after an entity the explanation names it 7.5% of the time, while a linear probe on the same activation still reads it 35% of the time. 22% of explanations name a wrong entity of the same type. |
| Why | Editing a wrong name out of an explanation changes the reconstruction score by about 0.002. The training objective has no reason to remove it. |
| What helps | Sample four explanations and keep the names at least two agree on: 74% of wrong names go, 77% of true names stay, with no training. Most of that signal is already in one explanation's token probabilities. |
| A test of the "why" | Fine-tuning on reconstruction alone changes nothing. Adding a cost on unsupported names cuts wrong names by 5 to 8% in two independent runs. Rewarding the true name does not bring it back. |

## Where to find each result

Every script is at the top level and is named after what it does. Each writes one markdown file of tables.

| Report chapter | Script | Result |
|---|---|---|
| 3 Setup: passages | `build_passages.py` | `runs/passages/` |
| 3 Setup: pipeline check | `verify_pipeline.py` | `runs/kaggle_generation/verify.log` |
| 4 Fading: probe and what explanations name | `fade_curve.py` | `runs/analysis/fade_curve.md` |
| 4 and 6: wrong names, positions, agreement, filter | `name_analysis.py` | `runs/analysis/name_analysis.md` |
| 5 Incentives: the edit test | `edit_test.py` | `runs/analysis/edit_test.md` |
| 6 Agreement: token confidence | `confidence.py` | `runs/analysis/confidence.md` |
| 7 Fine-tuning: training | `finetune.py` + `configs/` | `runs/finetune/<run>/` |
| 7 Fine-tuning: held-out evaluation | `evaluate.py` | `runs/finetune/eval/` |
| 7 Fine-tuning: comparison with the released verbalizer | `compare.py` | `runs/finetune/compare.md` |
| All tables and figures in the report | `paper/build_assets.py` | `paper/tables/`, `paper/figs/` |

Shared pieces: `nla_local.py` runs the three models (extract an activation, write an explanation, rebuild the
activation), `entities.py` holds the 60 entities and the statistics helpers, and `names.py` is the name
detector.

The 8,640 explanations everything is measured on are in `runs/analysis/nla-95f715bcdb.jsonl`.

### Fine-tuning runs

| Folder in `runs/finetune/` | Score used to choose training samples | Name in the report |
|---|---|---|
| `full-recon` | reconstruction cosine | Reconstruction |
| `full-lineup` | pick the activation out of four near-twins | Lineup |
| `full-holdback` | cosine minus a cost per unsupported name | Hold back |
| `seed1-holdback` | the same, from a different seed | Hold back, second run |
| `full-recall` | hold back plus a bonus for the true name | Keep the specific |
| `ep5-holdback` | hold back, one round of five passes | mentioned in section 7.4 |

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
git clone --depth 1 https://github.com/kitft/natural_language_autoencoders nla_ref
python fetch_artifacts.py        # large files from Hugging Face, about 650 MB
```

## Reproducing

These read the saved explanations and activations and need no GPU:

```bash
python fade_curve.py
python name_analysis.py
python edit_test.py
python confidence.py
python compare.py full-recon-round2 full-holdback-round2 seed1-holdback-round2 ep5-holdback-round0 full-lineup-round2
python paper/build_assets.py && (cd paper && tectonic main.tex)
```

These need a GPU (about 20 GB for inference at batch 128; fine-tuning peaked at 44 GB):

```bash
python finetune.py configs/holdback.json --n_train 2000 --k 8 --rounds 3 --tag full --gen_batch 128
python evaluate.py runs/finetune/full-holdback/round2/adapter --gen_batch 128
```

## Large files

Too big for git, so they live on Hugging Face and `fetch_artifacts.py` puts them in place:

- Fine-tuned adapters: <https://huggingface.co/tanvincible/nla-qwen2.5-7b-L20-av-specifics-lora>
- Activations, caches and training samples: <https://huggingface.co/datasets/tanvincible/nla-specifics-artifacts>

The models are the original authors' releases: `Qwen/Qwen2.5-7B-Instruct`, `kitft/nla-qwen2.5-7b-L20-av`
and `kitft/nla-qwen2.5-7b-L20-ar`.

## Other folders

- `infra/` holds the Kaggle job that generated the explanations (`infra/kaggle/kernel/generate_explanations.py`) and two shell scripts for rented GPUs. None
  of it is needed to check the results.
- `data/words.txt` is a word list used by the name detector.

## One thing to know before using the released NLA

The usage example published with the models reads `hidden_states[20]`. The NLA's "layer 20" is the output of
block 20, which is `hidden_states[21]` in Hugging Face indexing. Reading one layer early drops the
reconstruction cosine from 0.89 to 0.85 without any error. `nla_local.layer()` handles this.

## Credit

Built on [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) by
Fraser-Taliente, Kantamneni, Ong et al. (2026), who released the models and training code.
