#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 /path/to/model.gguf [threads] [ubatch] [ctx] }"
THREADS="${2:-$(nproc)}"
UBATCH="${3:-1024}"
CTX="${4:-8192}"
SEED="${SEED:-42}"
BIN="${BIN:-./build/bin/gptoss-cli}"
MODE="${MODE:-ht}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if ! [[ $THREADS =~ ^[0-9]+$ ]]; then
  echo "[bench] error: THREADS must be an integer (got '$THREADS')" >&2
  exit 1
fi

if ! [[ $UBATCH =~ ^[0-9]+$ ]]; then
  echo "[bench] error: UBATCH must be an integer (got '$UBATCH')" >&2
  exit 1
fi

if ! [[ $CTX =~ ^[0-9]+$ ]]; then
  echo "[bench] error: CTX must be an integer (got '$CTX')" >&2
  exit 1
fi

if ! [[ $SEED =~ ^-?[0-9]+$ ]]; then
  echo "[bench] error: SEED must be an integer (got '$SEED')" >&2
  exit 1
fi

if [[ "$MODE" == "phys" ]]; then
  if PHYS_MASK=$($SCRIPT_DIR/cpu_pinning.sh); then
    if [[ -z "$PHYS_MASK" ]]; then
      echo "[bench] failed to determine physical core mask; falling back to HT" >&2
      MODE=ht
    else
      DECODE_THREADS=$(awk -F, '{print NF}' <<<"$PHYS_MASK")
      TASKSET_MASK="$PHYS_MASK"
      echo "[bench] using physical primaries: $PHYS_MASK (decode_threads=$DECODE_THREADS)"
    fi
  else
    echo "[bench] cpu_pinning.sh failed; falling back to HT" >&2
    MODE=ht
  fi
fi

if [[ "$MODE" != "phys" ]]; then
  DECODE_THREADS="$THREADS"
  TASKSET_MASK="0-$((THREADS-1))"
  echo "[bench] using HT: mask=$TASKSET_MASK (threads=$DECODE_THREADS)"
fi

if [[ -z "${DECODE_THREADS:-}" || $DECODE_THREADS -le 0 ]]; then
  echo "[bench] error: unable to determine decode thread count" >&2
  exit 1
fi

MLOCK_FLAG=""
if [[ "${ENABLE_MLOCK:-0}" == "1" ]]; then
  MLOCK_FLAG="--mlock"
fi

cmd=("$BIN" -m "$MODEL" --measure-tps -t "$DECODE_THREADS" -tb "$THREADS" --ubatch-size "$UBATCH")
if [[ -n "$MLOCK_FLAG" ]]; then
  cmd+=("$MLOCK_FLAG")
fi
cmd+=(--numa none --ctx-size "$CTX" --seed "$SEED")

ts=$(date -u +%Y%m%dT%H%M%SZ)
logdir="tools/bench/logs"
mkdir -p "$logdir"
log="$logdir/tps-$ts.log"

echo "[bench] MODEL=$MODEL THREADS=$THREADS UBATCH=$UBATCH CTX=$CTX SEED=$SEED MODE=$MODE" | tee "$log"
set -x
numactl --cpunodebind=0 --membind=0 taskset -c "$TASKSET_MASK" env \
  OMP_NUM_THREADS="$DECODE_THREADS" \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  "${cmd[@]}" | tee -a "$log"
