#!/usr/bin/env bash
set -euo pipefail

export GPTOSS_QGEMV_DEBUG=1
export GPTOSS_FLASH_DEBUG=1
export GPTOSS_FLASH_DECODE=1
export GPTOSS_FLASH_TILE=${GPTOSS_FLASH_TILE:-256}

# build
cmake --build --preset build --target ggml-cpu

# run one single-token decode on a quantized model (adjust paths/binary)
# We expect exactly one line per fastpath:
#   [qgemv] AVX2 Q4_K decode fastpath enabled (n=1)   or MXFP4
#   [flash-decode] online-softmax fastpath enabled (tile=...)
./your_infer_binary --model path/to/model-q4_k_or_mxfp4.gguf \
    --prompt "Hello" --max-new-tokens 1 --threads 16
