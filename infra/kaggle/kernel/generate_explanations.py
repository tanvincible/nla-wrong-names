"""Kaggle job (GPU T4 x2, internet on, private):

  1. verify_pipeline.py: the exact Kaggle-recipe pipeline check. If it fails, stop before spending the long run.
  2. Generation: the released verbalizer explains the Mac-extracted activations (540 test passages x 4 distances,
     4 samples each), so it explains exactly the activations the probe was scored on.

The verbalizer runs in fp16 here (a T4 has no bf16), split across the two GPUs; the activations it reads were
extracted on the Mac in bf16. Explanations are appended to live-*.jsonl after every batch of 32.
Input: the private dataset holding generation_input-*.pt, the project .py files and the neutral-passage file.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
PROJ = WORK / "project"
REPOS = dict(target="Qwen/Qwen2.5-7B-Instruct", av="kitft/nla-qwen2.5-7b-L20-av", ar="kitft/nla-qwen2.5-7b-L20-ar")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, check=True):
    say(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, executable="/bin/bash")


def scratch_dir():
    """Largest writable disk outside /kaggle/working (whose 20 GB are saved as output)."""
    best, free = None, -1
    for c in ["/kaggle/tmp", "/kaggle/temp", "/tmp", "/root"]:
        try:
            Path(c).mkdir(parents=True, exist_ok=True)
            f = shutil.disk_usage(c).free
        except OSError:
            continue
        if f > free:
            best, free = Path(c), f
    say(f"scratch dir {best} with {free / 2**30:.0f} GiB free")
    return best, free


def assemble(live, rows, n, out):
    """live-*.jsonl (batches of explanations in order) -> one line per fully explained activation."""
    exps = {}
    if live.exists():
        for j in map(json.loads, live.read_text().splitlines()):
            for k, e in enumerate(j["explanations"]):
                exps[j["offset"] + k] = e
    done = 0
    with out.open("w") as f:
        for r, (i, d) in enumerate(rows):
            s = [exps.get(r * n + k) for k in range(n)]
            if None not in s:
                f.write(json.dumps({"i": i, "d": d, "samples": s}) + "\n")
                done += 1
    say(f"assembled {done}/{len(rows)} activations into {out.name}")
    return done


def main():
    inp = next(Path("/kaggle/input").rglob("generation_input-*.pt"))
    tag = inp.stem.split("-", 1)[1]
    tmp, free = scratch_dir()
    os.environ.update(HF_HOME=str(tmp / "hf"), NLA_DTYPE="float16", NLA_DEVICE_MAP="balanced", NLA_BATCH="32",
                      TOKENIZERS_PARALLELISM="false")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    import torch
    assert torch.cuda.device_count() == 2, f"expected GPU T4 x2, found {torch.cuda.device_count()} GPU(s)"
    assert free > 30 * 2**30, f"need at least 30 GiB of scratch, found {free / 2**30:.0f} GiB"
    low_disk = free < 60 * 2**30  # the three checkpoints are ~41 GiB together

    sh(f'{sys.executable} -m pip install -q "transformers>=4.56,<5" "huggingface-hub<1.0" accelerate safetensors '
       f'pyyaml orjson httpx pyarrow')
    PROJ.mkdir(parents=True, exist_ok=True)
    for f in inp.parent.glob("*.py"):
        shutil.copy(f, PROJ)
    (PROJ / "runs" / "passages").mkdir(parents=True, exist_ok=True)
    for f in inp.parent.glob("passages-*.json"):
        shutil.copy(f, PROJ / "runs" / "passages")
    if not (PROJ / "nla_ref").exists():
        sh(f"git clone --depth 1 -q https://github.com/kitft/natural_language_autoencoders {PROJ / 'nla_ref'}")
    os.chdir(PROJ)
    sys.path.insert(0, str(PROJ))

    # 1. Kaggle-recipe verification; its own exit code says pass or fail
    if low_disk:  # verify_pipeline.py is done with the target model once it starts loading the verbalizer: free it then
        def janitor():
            log_file = PROJ / "runs" / "progress.log"
            while True:
                if log_file.exists() and f"loading {REPOS['av']}" in log_file.read_text():
                    shutil.rmtree(tmp / "hf" / "hub" / ("models--" + REPOS["target"].replace("/", "--")), ignore_errors=True)
                    say("low disk: deleted the target model's cache")
                    return
                time.sleep(5)
        threading.Thread(target=janitor, daemon=True).start()
    t0 = time.time()
    v = sh(f"set -o pipefail; {sys.executable} -u verify_pipeline.py 2>&1 | tee {WORK / 'verify.log'}", check=False)
    say(f"verify_pipeline.py finished in {(time.time() - t0) / 60:.1f} min with exit code {v.returncode}")
    for f in (PROJ / "runs" / "passages").glob("verify-*.pt"):
        shutil.copy(f, WORK)
    if v.returncode != 0:
        raise SystemExit("verification failed: not starting generation")
    for r in ("target", "ar"):  # only the verbalizer is needed from here on
        shutil.rmtree(tmp / "hf" / "hub" / ("models--" + REPOS[r].replace("/", "--")), ignore_errors=True)

    # 2. generation
    from nla_local import DT, verbalize
    d = torch.load(inp)
    rows, n = [tuple(r) for r in d["rows"]], d["n"]
    live, out = WORK / f"live-{tag}.jsonl", WORK / f"nla-{tag}.jsonl"
    say(f"generation: {len(rows)} activations x {n} samples = {len(rows) * n} explanations, dtype {DT}, batch 32")
    t0 = time.time()
    try:
        verbalize(d["vectors"].float(), n=n, temperature=d["temperature"], batch=32, seed=0, live=str(live))
    finally:  # keep whatever was produced, even after a failure
        done = assemble(live, rows, n, out)
        mins = (time.time() - t0) / 60
        (WORK / f"meta-{tag}.json").write_text(json.dumps({
            "verbalizer_dtype": str(DT), "device_map": os.environ["NLA_DEVICE_MAP"], "batch": 32, "seed": 0,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "torch": torch.__version__, "activations_done": done, "activations_total": len(rows),
            "minutes": round(mins, 1), "explanations_per_minute": round(done * n / max(mins, 1e-9), 1),
            "note": "activations were extracted on the Mac in bf16; the verbalizer ran here in fp16"}, indent=1))
        if (PROJ / "runs" / "progress.log").exists():
            shutil.copy(PROJ / "runs" / "progress.log", WORK / "progress.log")
        shutil.rmtree(PROJ, ignore_errors=True)  # keep the saved output small
        say(f"generation took {mins:.1f} min")


if __name__ == "__main__":
    main()
