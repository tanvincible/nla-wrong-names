#!/usr/bin/env bash
# Run on the RunPod GPU box from the project folder: bash pod_run.sh
# Order: cross-check the pipeline, measure throughput, then H, used.py, step 4 (fade.py) and step 5 (used2.py).
# Every stage logs to runs/progress.log and runs/logs/; each script caches, so rerunning resumes.
set -uo pipefail
cd "$(dirname "$0")"
export NLA_BATCH="${NLA_BATCH:-64}" HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
mkdir -p runs/logs

pip install -q "transformers<5" "huggingface-hub<1.0" accelerate safetensors httpx orjson pyyaml numpy \
    scikit-learn matplotlib pyarrow 2>&1 | tail -2
[ -d nla_ref ] || git clone --depth 1 -q https://github.com/kitft/natural_language_autoencoders.git nla_ref
for r in kitft/nla-qwen2.5-7b-L20-av kitft/nla-qwen2.5-7b-L20-ar Qwen/Qwen2.5-7B-Instruct; do
    huggingface-cli download "$r" > /dev/null || { echo "download failed: $r"; exit 1; }
done

run() {  # run <script> [args]: keep going if one script fails, but say so
    echo "=== $(date +%H:%M:%S) $*" | tee -a runs/progress.log
    python "$@" 2>&1 | grep --line-buffered -v "Fetching\|Loading checkpoint\|torch_dtype\|Some weights\|You should probably" \
        | tee "runs/logs/pod-$(basename "$1" .py).log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || echo "=== FAILED: $*" | tee -a runs/progress.log
}
run verify.py --throughput
run planted.py          # planted-concept checks, rerun at the corrected layer
run checks_ar.py
run checks_target.py
run check_h.py
run used.py
run fade.py
run used2.py
echo "=== $(date +%H:%M:%S) all done" | tee -a runs/progress.log
