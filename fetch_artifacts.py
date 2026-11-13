"""Download the files that are too large for git into the places the scripts expect them (about 650 MB).

    python fetch_artifacts.py             activations, caches, training samples and the fine-tuned adapters
    python fetch_artifacts.py --adapters  only the adapters (about 40 MB each)

They live in two Hugging Face repos. While those are private, set HF_TOKEN to a token that can read them.
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODEL = "tanvincible/nla-qwen2.5-7b-L20-av-specifics-lora"  # one folder per scoring rule
DATA = "tanvincible/nla-specifics-artifacts"               # files at the same paths they have in this repo
ADAPTERS = {"recon": "full-recon/round2", "holdback": "full-holdback/round2", "lineup": "full-lineup/round2",
            "holdback-5epochs": "ep5-holdback/round0", "recall": "full-recall/round2"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", action="store_true")
    ap.add_argument("--root", default=str(Path(__file__).parent))
    a = ap.parse_args()
    root = Path(a.root)
    if not a.adapters:
        snapshot_download(DATA, repo_type="dataset", local_dir=root, allow_patterns=["runs/*"])
        print("data files are under", root / "runs")
    for name, run in ADAPTERS.items():
        try:
            src = hf_hub_download(MODEL, f"{name}/adapter_model.safetensors")
        except Exception as e:  # an arm that was never uploaded should not stop the others
            print(f"adapter {name}: not available ({type(e).__name__})")
            continue
        dst = root / "runs" / "hb_pod" / run / "adapter" / "adapter_model.safetensors"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"adapter {name} -> {dst}")
