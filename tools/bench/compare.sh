#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-models/gpt-oss-20b-Q4_K_M.gguf}
UBATCH=${2:-1024}
CTX=${3:-8192}
SEED=${4:-42}
BIN="${BIN:-./build/bin/gptoss-cli}"

run_one() {
  local threads=$1
  local log
  log=$(mktemp)
  local cmd
  cmd=$(cat <<EOF_CMD
env \
  OMP_NUM_THREADS=$threads \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  GOMP_CPU_AFFINITY=0-$((threads-1)) \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  "$BIN" \
    -m "$MODEL" \
    --measure-tps \
    -t $threads \
    -tb $threads \
    --ubatch-size $UBATCH \
    --numa none \
    --ctx-size $CTX \
    --seed $SEED
EOF_CMD
  )
  numactl --cpunodebind=0 --membind=0 taskset -c 0-$((threads-1)) bash -lc "$cmd" | tee "$log" >/dev/null
  awk '/TPS:/ {t=$NF} END {print t+0}' "$log"
  rm -f "$log"
}

t12=$(run_one 12)
t16=$(run_one 16)

printf "TPS @12: %s\n" "$t12"
printf "TPS @16: %s\n" "$t16"

better=12
best=$t12
if awk "BEGIN{exit !($t16 > $t12)}"; then
  better=16
  best=$t16
fi

printf "→ Best: %s threads (TPS=%s)\n" "$better" "$best"
