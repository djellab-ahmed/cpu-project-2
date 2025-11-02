#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: baseline.sh <model.gguf> [threads] [ubatch] [ctx] [seed]}"
THREADS="${2:-16}"
UBATCH="${3:-1024}"
CTX="${4:-8192}"
SEED="${5:-42}"

echo "[bench] MODEL=$MODEL THREADS=$THREADS UBATCH=$UBATCH CTX=$CTX SEED=$SEED"

# MODE controls how we pin:
#   MODE=ht   -> use all logical CPUs 0..THREADS-1 (HT on)
#   MODE=phys -> use first sibling per physical core on socket 0 (HT off)
MODE="${MODE:-ht}"

# Decide taskset mask and decode threads
if [[ "$MODE" == "phys" ]]; then
  PHYS_MASK="$(tools/bench/cpu_pinning.sh)"
  DECODE_THREADS="$(awk -F, '{print NF}' <<<"$PHYS_MASK")"
  TASKSET_MASK="$PHYS_MASK"
  echo "[bench] using physical primaries: $PHYS_MASK (count=$DECODE_THREADS)"
else
  DECODE_THREADS="$THREADS"
  TASKSET_MASK="0-$((THREADS-1))"
  echo "[bench] using HT: mask=$TASKSET_MASK (threads=$DECODE_THREADS)"
fi

LOG_DIR="tools/bench/logs"
mkdir -p "$LOG_DIR"
ts="$(date -u +"%Y%m%dT%H%M%SZ")"
LOG="$LOG_DIR/tps-$ts.log"
echo "[bench] saving full output to $LOG"

# Build the command (prefill uses -tb = THREADS; decode uses -t = DECODE_THREADS)
CMD="./build/bin/gptoss-cli \
  -m \"$MODEL\" \
  --measure-tps \
  -t $DECODE_THREADS \
  -tb $THREADS \
  --ubatch-size $UBATCH \
  --mlock \
  --numa none \
  --ctx-size $CTX \
  --seed $SEED"

# Warmup (optional, speeds up first run); ignore failures
./build/bin/gptoss-cli -m "$MODEL" -p warmup -n 64 -t "$DECODE_THREADS" -tb "$THREADS" --ubatch-size "$UBATCH" --numa none >/dev/null 2>&1 || true

# Run pinned to a single NUMA node and the exact cpu list/range
numactl --cpunodebind=0 --membind=0 \
taskset -c "$TASKSET_MASK" \
bash -lc '
  env \
    OMP_NUM_THREADS='"'"'$DECODE_THREADS'"'"' \
    OMP_DYNAMIC='"'"'${OMP_DYNAMIC:-TRUE}'"'"' \
    OMP_PROC_BIND=close \
    OMP_PLACES=cores \
    OMP_WAIT_POLICY=PASSIVE \
    GOMP_CPU_AFFINITY="'"'"$TASKSET_MASK"'"'" \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
  '"'"$CMD"'"'
' | tee -a "$LOG"
