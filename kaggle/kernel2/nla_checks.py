"""Kaggle job (GPU T4 x2, internet on, private): the checks that were queued behind the fade study.

Runs, in order, each as its own script with its own log:
  planted.py         plant a known concept in an activation; is it named?          (planted)
  checks_ar.py       wrong-name cost under the reconstruction score, and related    (planted)
  checks_target.py   verify against the activation, model-prediction score, edits   (planted)
  check_h.py         explain what is left after subtracting the theme               (planted)
  used.py            12 prompts where the model is about to say the concept         (natural prompts)
  used2.py           120 quiz questions and 12 couplets, with a logit-lens baseline (natural prompts)

All at the corrected layer (output of block 20). Everything runs in fp16 here (a T4 has no bf16), with each
model split across the two GPUs. A failing script does not stop the ones after it; results are copied to the
kernel output after every script.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
PROJ = Path("/kaggle/tmp/project")
SCRIPTS = ["planted.py", "checks_ar.py", "checks_target.py", "check_h.py", "used.py", "used2.py"]


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, check=True):
    say(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, executable="/bin/bash")


def save():
    """Copy logs and results (not model caches) into the kernel output."""
    if (PROJ / "runs").exists():
        shutil.copytree(PROJ / "runs", WORK / "runs", dirs_exist_ok=True)


def main():
    src = next(Path("/kaggle/input").rglob("used2.py")).parent
    os.environ.update(HF_HOME="/kaggle/tmp/hf", NLA_DTYPE="float16", NLA_DEVICE_MAP="balanced", NLA_BATCH="32",
                      TOKENIZERS_PARALLELISM="false")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    import torch
    assert torch.cuda.device_count() == 2, f"expected GPU T4 x2, found {torch.cuda.device_count()} GPU(s)"
    sh(f'{sys.executable} -m pip install -q "transformers>=4.56,<5" "huggingface-hub<1.0" accelerate safetensors '
       f'pyyaml orjson httpx')
    PROJ.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.py"):
        shutil.copy(f, PROJ)
    if not (PROJ / "nla_ref").exists():
        sh(f"git clone --depth 1 -q https://github.com/kitft/natural_language_autoencoders {PROJ / 'nla_ref'}")
    (PROJ / "runs" / "logs").mkdir(parents=True, exist_ok=True)
    os.chdir(PROJ)
    status = {}
    for s in SCRIPTS:
        t0 = time.time()
        r = sh(f"set -o pipefail; {sys.executable} -u {s} 2>&1 | grep --line-buffered -v 'Fetching\\|Loading checkpoint' "
               f"| tee runs/logs/{Path(s).stem}.log", check=False)
        status[s] = f"exit {r.returncode}, {(time.time() - t0) / 60:.1f} min"
        say(f"{s}: {status[s]}")
        save()
    say("summary: " + "; ".join(f"{k} {v}" for k, v in status.items()))
    (WORK / "status.txt").write_text("\n".join(f"{k}: {v}" for k, v in status.items())
                                     + "\nall models ran in fp16 on 2x Tesla T4, corrected layer (output of block 20)\n")


if __name__ == "__main__":
    main()
