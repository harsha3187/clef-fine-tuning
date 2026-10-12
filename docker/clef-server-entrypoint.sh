#!/bin/sh
# Check the model folder, then serve it with the same HTTP server Azure ML uses. POST /score, GET / for health.
set -eu

model_dir="${CLEF_MODEL_DIR:-/models}"
echo "checking the model in $model_dir"
if ! PYTHONPATH=/app/code python -m routing_lab.verify_model "$model_dir"; then
  echo "The model folder is not usable (see above). Mount models/clef-flash-capability-ft-v3 at $model_dir, and run 'git lfs pull' if the weights are pointer files." >&2
  exit 1
fi

# Fail now, not after downloading 18 GB of base model and then finding no GPU.
device="${CLEF_DEVICE:-cuda}"
if [ "$device" = "cuda" ] && ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "No CUDA GPU is visible to this container." >&2
  echo "  Run it on a GPU host with the NVIDIA Container Toolkit (docker run --gpus all ..., or the compose profile local-clef)." >&2
  echo "  Or set CLEF_DEVICE=cpu to run without one: it needs about 20 GB of memory and takes tens of seconds per request." >&2
  exit 1
fi
echo "device: $device; the first start downloads the base model into ${HF_HOME:-/cache/huggingface}"
exec azmlinfsrv --entry_script /app/code/score.py --port 5001 "$@"