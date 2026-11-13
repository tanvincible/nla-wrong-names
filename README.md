# Wrong Names in Natural Language Autoencoder Explanations

Course project for CS787 (Generative Artificial Intelligence), IIT Kanpur.

A natural language autoencoder (NLA) explains an activation inside a language model by writing a paragraph
about it. The paragraphs read well, but they sometimes name a specific that was never in the text, for
example "piano" when the passage said "cello". This project measures how often that happens on the released
NLA for Qwen2.5-7B, shows why the training objective allows it, and compares two ways of removing it.

**The report is [`paper/main.pdf`](paper/main.pdf).**

## Findings

1. Sixteen tokens after an entity, the explanation names it 7.5% of the time; a linear probe on the same
   activations still reads it 35% of the time; 22% of explanations contain a wrong name of the same type.
2. Editing a wrong name out of an explanation changes the reconstruction score by about 0.002, so training
   on reconstruction has no reason to remove it.
3. Keeping only the names that at least 2 of 4 sampled explanations agree on removes 74% of wrong names and
   keeps 77% of true ones, with no training.
4. Fine-tuning with a penalty on unsupported names cuts wrong names by 8% on held-out passages; fine-tuning
   on reconstruction alone changes nothing.

## Where things are

| Path | What it is |
|---|---|
| `paper/` | The report (LaTeX source, PDF) and `build_assets.py`, which builds every table and figure from the saved results |
| `runs/fade2/` | Main measurements: tables, the 8,640 explanations, follow-up analyses, the edit test |
| `runs/hb_pod/` | Fine-tuning: configs, per-round histories, kept samples, held-out evaluations |
| `runs/hb/compare.md` | Each fine-tuned verbalizer against the released one, paired on the same passages |
| `runs/kaggle_step4/`, `runs/kaggle_checks/` | Outputs of the two Kaggle GPU jobs |
| `runs_layer20_offbyone/` | Early results from before the layer fix; kept for the record, not used |

Large files are on Hugging Face and are fetched by `fetch_artifacts.py`:

- Fine-tuned adapters: <https://huggingface.co/tanvincible/nla-qwen2.5-7b-L20-av-specifics-lora>
- Activations, caches and training samples: <https://huggingface.co/datasets/tanvincible/nla-specifics-artifacts>

The models themselves are the authors' releases: `Qwen/Qwen2.5-7B-Instruct`, `kitft/nla-qwen2.5-7b-L20-av`
and `kitft/nla-qwen2.5-7b-L20-ar`.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
git clone --depth 1 https://github.com/kitft/natural_language_autoencoders nla_ref
python fetch_artifacts.py        # about 650 MB; set HF_TOKEN while the Hugging Face repos are private
```

## Reproducing the results

No GPU needed (reads saved explanations and activations):

```bash
python fade2.py                                    # fade curve, probe, what the explanations name
python step4_extra.py                              # agreement, filter, name position, fair probe comparison
python task1_edits.py                              # the edit test (uses the cached reconstructions)
python hb_compare.py full-recon-round2 full-holdback-round2 full-lineup-round2
python paper/build_assets.py                       # tables and figures for the report
cd paper && tectonic main.tex                      # the PDF
```

GPU needed (about 20 GB for inference at batch 128; fine-tuning peaked at 44 GB):

```bash
python hb.py configs/holdback.json --n_train 2000 --k 8 --rounds 3 --tag full --gen_batch 128
python hb_eval.py runs/hb/full-holdback/round2/adapter --gen_batch 128
```

`kaggle/` holds the scripts that generated the explanations on Kaggle, and `pod_bootstrap.sh` sets up a
rented GPU from scratch.

## One thing to know before using the released NLA

The usage example published with the models reads `hidden_states[20]`. The NLA's "layer 20" is the output of
block 20, which is `hidden_states[21]` in Hugging Face indexing. Reading one layer early drops the
reconstruction cosine from 0.89 to 0.85 without any error. `nla_local.layer()` handles this.

## Credit

Built on [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html) by
Fraser-Taliente, Kantamneni, Ong et al. (2026), who released the models and training code.
