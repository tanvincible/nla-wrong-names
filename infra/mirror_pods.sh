#!/bin/bash
# Pull training results from every pod listed in runs/pods.txt (name ip port) to this Mac every 3 minutes,
# so nothing is lost if a pod stops or credits run out. Reconstruction caches are skipped: they are large and recomputable.
cd "$(dirname "$0")/.."
while true; do
  while read -r name ip port; do
    [ -z "$name" ] && continue
    rsync -az --timeout=60 -e "ssh -i $HOME/.ssh/runpod_ed25519 -p $port -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new" \
      --exclude 'recs.pt' --exclude '*-recs.pt' "root@$ip:/workspace/project/runs/finetune/" "runs/finetune/" 2>/dev/null \
      && rsync -az --timeout=60 -e "ssh -i $HOME/.ssh/runpod_ed25519 -p $port -o ConnectTimeout=15" "root@$ip:/workspace/full-*.log" "runs/finetune/logs-$name/" 2>/dev/null
  done < runs/pods.txt
  date +"%H:%M:%S mirrored" > runs/finetune/last_mirror.txt
  sleep 180
done
