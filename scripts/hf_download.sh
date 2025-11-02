#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
hf download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf --local-dir ./models
echo "models/gpt-oss-20b-Q4_K_M.gguf"
