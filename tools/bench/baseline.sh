#!/usr/bin/env bash
set -euo pipefail

# Required: path to GGUF model
MODEL="${1:?USAGE: tools/bench/baseline.sh /path/to/model.gguf [THREADS] [UBATCH] [CTX]}"

THREADS="${2:-$(nproc)}"    # default: all cores (e.g., 16)
UBATCH="${3:-1024}"         # prompt micro-batch size
CTX="${4:-8192}"            # context size
SEED="${SEED:-42}"          # fixed seed for reproducibility

# Pin to a single NUMA node and cores [0..THREADS-1]
# Also make OpenMP & any BLAS libs single-thread consistent with THREADS
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

CMD="./build/bin/gptoss-cli \
  -m \"$MODEL\" \
  --measure-tps \
  -t $THREADS \
  -tb $THREADS \
  --ubatch-size $UBATCH \
  --mlock \
  --numa distribute \
  --ctx-size $CTX \
  --seed $SEED"

# Ensure logs directory exists
mkdir -p tools/bench/logs
STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
LOG="tools/bench/logs/tps-$STAMP.log"

echo "[bench] MODEL=$MODEL THREADS=$THREADS UBATCH=$UBATCH CTX=$CTX SEED=$SEED"
echo "[bench] saving full output to $LOG"

# Single NUMA node + explicit core pinning
set -x
numactl --cpunodebind=0 --membind=0 \
taskset -c 0-$((THREADS-1)) \
bash -lc "$CMD" | tee "$LOG"
set +x

# Surface the final True TPS line for convenience
echo
echo "[bench] summary:"
grep -E 'True TPS|Overall|TOTAL' -n "$LOG" || true
