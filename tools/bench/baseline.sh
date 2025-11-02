#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 /path/to/model.gguf [threads] [ubatch] [ctx] }"
THREADS="${2:-$(nproc)}"
UBATCH="${3:-1024}"
CTX="${4:-8192}"
SEED="${SEED:-42}"
BIN="${BIN:-./build/bin/gptoss-cli}"

ts=$(date -u +%Y%m%dT%H%M%SZ)
logdir="tools/bench/logs"
mkdir -p "$logdir"
log="$logdir/tps-$ts.log"

echo "[bench] MODEL=$MODEL THREADS=$THREADS UBATCH=$UBATCH CTX=$CTX SEED=$SEED" | tee "$log"

PHYS=$(tools/bench/cpu_pinning.sh)
NPHYS=$(( $(echo "$PHYS" | tr -cd ',' | wc -c) + 1 ))

# Prefer decode_threads = min(requested THREADS, physical cores)
DECODE_THREADS=${THREADS}
if [ "$THREADS" -gt "$NPHYS" ]; then
  DECODE_THREADS=$NPHYS
fi

echo "[bench] using physical cores: $PHYS (count=$NPHYS)  decode_threads=$DECODE_THREADS  prefill_threads=$THREADS"

numactl --cpunodebind=0 --membind=0 \
        taskset -c "$PHYS" \
        bash -lc 'env \
  OMP_NUM_THREADS='"$DECODE_THREADS"' \
  OMP_PROC_BIND=TRUE \
  OMP_PLACES=threads \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  ./build/bin/gptoss-cli \
    -m '"$MODEL"' \
    --measure-tps \
    -t '"$DECODE_THREADS"' \
    -tb '"$THREADS"' \
    --ubatch-size '"$UBATCH"' \
    --mlock \
    --numa none \
    --ctx-size '"$CTX"' \
    --seed '"$SEED"'' | tee -a "$logdir/tps-$ts.log"
