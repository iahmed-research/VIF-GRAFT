#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-benchmark}"
raw="${GRAFT_RAW:-data/vtac-1.0}"
out="${GRAFT_OUT:-outputs/reproduction}"
mkdir -p "$out"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$mode" in
  benchmark)
    graft doctor --device cuda
    graft prepare --raw "$raw" --out "$out" --device cuda 2>&1 | tee "$out/gcp-prepare.log"
    graft benchmark --raw "$raw" --out "$out" --device cuda 2>&1 | tee "$out/gcp-benchmark.log"
    ;;
  one-epoch)
    graft train --family QMF --fold 0 --seed 20260911 --epoch-limit 1 --raw "$raw" --out "$out" --device cuda 2>&1 | tee -a "$out/gcp-training.log"
    ;;
  all)
    graft all --raw "$raw" --out "$out" --device cuda 2>&1 | tee -a "$out/gcp-full.log"
    ;;
  *) printf 'Use benchmark, one-epoch or all\n' >&2; exit 2 ;;
esac
