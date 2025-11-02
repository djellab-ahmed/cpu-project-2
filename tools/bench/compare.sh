#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?usage: compare.sh <model.gguf> [ubatch] [ctx]}"
UBATCH="${2:-1024}"
CTX="${3:-8192}"

run() {
  local mode="$1" thr="$2"
  OMP_DYNAMIC=TRUE MODE="$mode" ./tools/bench/baseline.sh "$MODEL" "$thr" "$UBATCH" "$CTX" 42 | awk '/TPS:/{print $NF; exit}'
}

echo "Warmup..."
./build/bin/gptoss-cli -m "$MODEL" -p warmup -n 64 -t 16 -tb 16 --ubatch-size "$UBATCH" --numa none >/dev/null 2>&1 || true

echo "MODE=ht  t=16  TPS=$(run ht 16)"
echo "MODE=ht  t=14  TPS=$(run ht 14)"
echo "MODE=ht  t=12  TPS=$(run ht 12)"
echo "MODE=phys(t=8) TPS=$(run phys 16)"
