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
set -x
numactl --cpunodebind=0 --membind=0 taskset -c 0-$((THREADS-1)) \
"$BIN" -m "$MODEL" --measure-tps \
  -t "$THREADS" -tb "$THREADS" \
  --ubatch-size "$UBATCH" --mlock --numa distribute --ctx-size "$CTX" --seed "$SEED" \
| tee -a "$log"
