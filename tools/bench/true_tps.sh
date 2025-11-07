#!/usr/bin/env bash
set -euo pipefail
: "${THREADS:=16}"
: "${CTX:=8192}"
: "${NPRED:=256}"    # use 256 per prompt to keep runtime reasonable
: "${MODEL:=models/gpt-oss-20b-Q4_K_M.gguf}"

PROMPTS_FILE="${1:-tools/bench/prompts.txt}"
BIN="${BIN:-./build/bin/gptoss-cli}"

KV_Q8_ARGS=()
if [[ "${KV_Q8:-0}" != "0" ]]; then
  KV_Q8_ARGS+=(--kv-q8 --kv-q8-scheme row,row)
fi

total_tok=0
total_sec=0
prompt_idx=0

while IFS= read -r P; do
  # the CLI should print a single CSV line at end in the form: "BENCH,P_MS=...,D_MS=...,TOK=..."
  OUT=$("$BIN" -m "$MODEL" -p "$P" -n "$NPRED" -t "$THREADS" --ctx-size "$CTX" "${KV_Q8_ARGS[@]}" --quiet --bench)
  # Expect last line like: BENCH,P_MS=...,D_MS=...,TOK=...
  LINE=$(echo "$OUT" | awk -F',' '/^BENCH/ {print $0}' | tail -n1)
  if [[ -z "$LINE" ]]; then
    echo "error: CLI output did not contain a BENCH line" >&2
    exit 1
  fi
  P_MS=$(echo "$LINE" | sed -n 's/.*P_MS=\([0-9.]*\).*/\1/p')
  D_MS=$(echo "$LINE" | sed -n 's/.*D_MS=\([0-9.]*\).*/\1/p')
  TOK=$(echo "$LINE" | sed -n 's/.*TOK=\([0-9]*\).*/\1/p')
  if [[ -z "$P_MS" || -z "$D_MS" || -z "$TOK" ]]; then
    echo "error: failed to parse BENCH line: $LINE" >&2
    exit 1
  fi
  SEC=$(python3 - <<PY
p=float("${P_MS}")/1000.0; d=float("${D_MS}")/1000.0
print(p+d)
PY
  )
  TOTAL_MS=$(python3 - <<PY
p=float("${P_MS}"); d=float("${D_MS}")
print(p+d)
PY
  )
  PROMPT_TPS=$(python3 - <<PY
tok=int("${TOK}")
sec=float("${SEC}")
print(f"{tok/sec:.2f}")
PY
  )
  total_tok=$(( total_tok + TOK ))
  total_sec=$(python3 - <<PY
import sys
a=float("${total_sec}") if "${total_sec}"!="" else 0.0
b=float("${SEC}")
print(a+b)
PY
  )
  prompt_idx=$(( prompt_idx + 1 ))
  printf 'PROMPT %02d: tokens=%s total_ms=%s prefill_ms=%s decode_ms=%s TPS=%s\n' \
    "$prompt_idx" "$TOK" "$TOTAL_MS" "$P_MS" "$D_MS" "$PROMPT_TPS"
done < "$PROMPTS_FILE"

python3 - <<PY
tok=int("${total_tok}")
sec=float("${total_sec}")
print(f"TRUE_TPS={tok/sec:.2f}  (tokens={tok}, seconds={sec:.3f})")
PY
