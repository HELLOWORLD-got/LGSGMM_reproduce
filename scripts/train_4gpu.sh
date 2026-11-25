#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  smoke)
    exec python -m torch.distributed.run --standalone --nproc_per_node=4 \
      -m lgsgmm.run smoke --config configs/mdmt_4gpu_smoke.json --device cuda "$@"
    ;;
  train)
    exec python -m torch.distributed.run --standalone --nproc_per_node=4 \
      -m lgsgmm.run train --config configs/mdmt_4gpu.json --device cuda "$@"
    ;;
  evaluate)
    exec python -m torch.distributed.run --standalone --nproc_per_node=4 \
      -m lgsgmm.run evaluate --config configs/mdmt_4gpu.json --device cuda "$@"
    ;;
  *)
    echo "Usage: bash scripts/train_4gpu.sh [smoke|train|evaluate] [arguments...]" >&2
    exit 2
    ;;
esac
