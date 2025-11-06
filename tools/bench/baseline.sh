#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-models/gpt-oss-20b-Q4_K_M.gguf}
THREADS=${2:-16}
NPRED=${3:-256}
PROMPTS=${4:-tools/bench/prompts10.txt}
OUT=${5:-tools/bench/logs/suite.json}

mkdir -p "$(dirname "$OUT")"
./build/bin/tps-suite \
  --cli ./build/bin/gptoss-cli \
  --model "$MODEL" \
  --threads "$THREADS" \
  --n-predict "$NPRED" \
  --prompts "$PROMPTS" \
  --out "$OUT"
jq . "$OUT" 2>/dev/null || cat "$OUT"
