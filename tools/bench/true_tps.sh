#!/usr/bin/env bash
set -euo pipefail
: "${THREADS:=16}"
: "${CTX:=8192}"
: "${NPRED:=256}"    # use 256 per prompt to keep runtime reasonable
: "${MODEL:=models/gpt-oss-20b-Q4_K_M.gguf}"

PROMPTS_FILE="${1:-tools/bench/prompts.txt}"
BIN="${BIN:-./build/bin/gptoss-cli}"

total_tok=0
total_sec=0

while IFS= read -r P; do
  # the CLI should print a single CSV line at end in the form: "BENCH,P_MS=...,D_MS=...,TOK=..."
  OUT=$("$BIN" -m "$MODEL" -p "$P" -n "$NPRED" -t "$THREADS" --ctx-size "$CTX" --quiet --bench)
  # Expect last line like: BENCH,P_MS=...,D_MS=...,TOK=...
  LINE=$(echo "$OUT" | awk -F',' '/^BENCH/ {print $0}' | tail -n1)
  P_MS=$(echo "$LINE" | sed -n 's/.*P_MS=\([0-9.]*\).*/\1/p')
  D_MS=$(echo "$LINE" | sed -n 's/.*D_MS=\([0-9.]*\).*/\1/p')
  TOK=$(echo "$LINE" | sed -n 's/.*TOK=\([0-9]*\).*/\1/p')
  SEC=$(python - <<PY
p=float("${P_MS}")/1000.0; d=float("${D_MS}")/1000.0
print(p+d)
PY
  )
  total_tok=$(( total_tok + TOK ))
  total_sec=$(python - <<PY
import sys
a=float("${total_sec}") if "${total_sec}"!="" else 0.0
b=float("${SEC}")
print(a+b)
PY
  )
done < "$PROMPTS_FILE"

python - <<PY
tok=int("${total_tok}")
sec=float("${total_sec}")
print(f"TRUE_TPS={tok/sec:.2f}  (tokens={tok}, seconds={sec:.3f})")
PY
