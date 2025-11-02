#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-models/gpt-oss-20b-Q4_K_M.gguf}
UB=${2:-1024}
CTX=${3:-8192}
CANDIDATES=("12" "14" "16")
best_t=""
best_tps=0
for t in "${CANDIDATES[@]}"; do
  out=$(./tools/bench/baseline.sh "$MODEL" "$t" "$UB" "$CTX" | sed -n 's/.*TPS: \([0-9.]\+\).*/\1/p' | head -n1)
  tps=${out:-0}
  printf "%2s threads -> TPS=%s\n" "$t" "$tps"
  awk "BEGIN{exit !($tps > $best_tps)}" && { best_tps=$tps; best_t=$t; }
done
echo ">>> Best: threads=$best_t  TPS=$best_tps"
