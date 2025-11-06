#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-models/gpt-oss-20b-Q4_K_M.gguf}"
export GPTOSS_KV_DEBUG=1
export GPTOSS_KV_INTERLEAVE=1
export GPTOSS_KV_HUGEPAGES=1
export GPTOSS_FLASH_DECODE=1
export GPTOSS_FLASH_DEBUG=1
./gptoss-cli --model "$MODEL" --decode 32 --ctx 4096 --seed 1 >/dev/null
echo "[ok] kv-layout + flash-decode banners printed once"
