#!/bin/bash
# In-container entrypoint for Vertex custom training. Pulls shards + (if resuming) the latest
# checkpoint from GCS to local disk, then runs the trainer (which checkpoints back to GCS).
# On a Spot preemption the job restarts and re-runs this: it re-pulls ckpt_latest.pt and resumes.
set -uo pipefail
CFG="${1:-configs/train_h100.yaml}"
BUCKET="gs://serblo-vertex-a9a36e4a/serblo"

echo "=== vertex_train START $(date -u) | cfg=$CFG ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo "(no nvidia-smi)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-CUDA')" 2>&1

mkdir -p /workspace/tokenized /workspace/ckpts

echo "--- pull shards ($(date -u)) ---"
gsutil -m -q cp "$BUCKET/tokenized/*" /workspace/tokenized/ 2>&1 | tail -3 \
  || gcloud storage cp "$BUCKET/tokenized/*" /workspace/tokenized/ 2>&1 | tail -3
echo "shards local: $(ls /workspace/tokenized/*.bin 2>/dev/null | wc -l) bin + meta $(test -f /workspace/tokenized/meta.json && echo yes || echo NO)"

echo "--- resume checkpoint? ($(date -u)) ---"
if gsutil -q cp "$BUCKET/checkpoints/ckpt_latest.pt" /workspace/ckpts/ckpt_latest.pt 2>/dev/null; then
  echo "RESUME: pulled ckpt_latest.pt ($(du -h /workspace/ckpts/ckpt_latest.pt | cut -f1))"
else
  echo "FRESH: no checkpoint in GCS"
fi

python3 -c "import yaml" 2>/dev/null || pip install -q pyyaml 2>&1 | tail -1

echo "--- TRAIN ($(date -u)) ---"
cd /workspace
PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m serblo.train.train --config "$CFG"
echo "=== vertex_train END $(date -u) rc=$? ==="
