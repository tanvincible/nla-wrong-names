#!/bin/bash
# usage: pod_bootstrap.sh <ip> <port>   Upload code and data to a fresh pod and install everything it needs.
set -uo pipefail
cd "$(dirname "$0")/.."
ip=$1; port=$2
S="ssh -i $HOME/.ssh/runpod_ed25519 -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
$S root@$ip 'mkdir -p /workspace/project /workspace/hf /workspace/nltk_data' < /dev/null
rsync -az -e "$S" --include='*.py' --include='configs/***' --include='data/***' --exclude='*' ./ root@$ip:/workspace/project/
rsync -azR -e "$S" runs/passages/passages-bb2077b20f.json runs/passages/passages-a5b4344c76.json runs/analysis/nla-95f715bcdb.jsonl runs/analysis/acts-243dbf9dd6.pt root@$ip:/workspace/project/
$S root@$ip 'cat > /workspace/env.sh <<EOT
export HF_HOME=/workspace/hf NLTK_DATA=/workspace/nltk_data TOKENIZERS_PARALLELISM=false NLA_BATCH=128 PYTHONUNBUFFERED=1
[ -f /workspace/venv/bin/activate ] && source /workspace/venv/bin/activate
EOT
cat > /workspace/setup.sh <<"EOT"
source /workspace/env.sh
[ -x /workspace/venv/bin/python ] || python -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate
pip install -q "transformers<5" "huggingface-hub<1.0" peft accelerate safetensors pyyaml orjson httpx spacy geonamescache nltk scikit-learn matplotlib 2>&1 | tail -2
pip install -q https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl 2>&1 | tail -1
python -c "import nltk; nltk.download(\"wordnet\", download_dir=\"/workspace/nltk_data\", quiet=True)"
[ -d /workspace/project/nla_ref ] || git clone --depth 1 -q https://github.com/kitft/natural_language_autoencoders /workspace/project/nla_ref
python -c "import torch, transformers, peft, spacy, nltk, geonamescache, sklearn; print(\"deps ok\", torch.__version__, torch.cuda.is_available(), transformers.__version__, peft.__version__)"
for r in kitft/nla-qwen2.5-7b-L20-av kitft/nla-qwen2.5-7b-L20-ar Qwen/Qwen2.5-7B-Instruct; do huggingface-cli download $r > /dev/null 2>&1 && echo "downloaded $r" || echo "FAILED $r"; done
echo SETUP_DONE
EOT
cat > /workspace/run_arm.sh <<"EOT"
#!/bin/bash
cd /workspace/project && source /workspace/env.sh && ulimit -n 65536
arm=$1; shift
nohup python finetune.py configs/$arm.json --n_train 2000 --k 8 --rounds 3 --tag full --gen_batch 128 "$@" > /workspace/full-$arm.log 2>&1 < /dev/null &
echo "started $arm pid $!"
EOT
chmod +x /workspace/run_arm.sh
(nohup bash /workspace/setup.sh > /workspace/setup.log 2>&1 < /dev/null &); echo "setup started on $(hostname)"' < /dev/null
