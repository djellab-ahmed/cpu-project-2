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
cmd=$(cat <<EOF
env \\
  OMP_NUM_THREADS=$THREADS \\
  OMP_PROC_BIND=close \\
  OMP_PLACES=cores \\
  GOMP_CPU_AFFINITY=0-$((THREADS-1)) \\
  OPENBLAS_NUM_THREADS=1 \\
  MKL_NUM_THREADS=1 \\
  BLIS_NUM_THREADS=1 \\
  "$BIN" \\
    -m "$MODEL" \\
    --measure-tps \\
    -t $THREADS \\
    -tb $THREADS \\
    --ubatch-size $UBATCH \\
    --mlock \\
    --numa none \\
    --ctx-size $CTX \\
    --seed $SEED
EOF
)
numactl --cpunodebind=0 --membind=0 taskset -c 0-$((THREADS-1)) bash -lc "$cmd" | tee -a "$log"
