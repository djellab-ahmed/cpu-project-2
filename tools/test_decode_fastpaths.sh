#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 MODEL_PATH [extra gptoss-cli args...]" >&2
    exit 1
fi

MODEL_PATH=$1
shift

BINARY=${BINARY:-./build/bin/gptoss-cli}

if [[ ! -x ${BINARY} ]]; then
    echo "error: expected gptoss-cli binary at '${BINARY}'." >&2
    echo "build it with: cmake --preset dev -S . -B build && cmake --build --preset build --target gptoss-cli" >&2
    exit 2
fi

export GPTOSS_QGEMV_DEBUG=1
export GPTOSS_FLASH_DEBUG=1
export GPTOSS_FLASH_DECODE=${GPTOSS_FLASH_DECODE:-1}
export GPTOSS_FLASH_TILE=${GPTOSS_FLASH_TILE:-256}

# build the CLI to ensure fast-path objects are linked
cmake --build --preset build --target gptoss-cli >/dev/null

tmp_log=$(mktemp)
trap 'rm -f "$tmp_log"' EXIT

"${BINARY}" \
    -m "${MODEL_PATH}" \
    --prompt "Hello" \
    --n-predict 1 \
    --threads 1 \
    "$@" 2>&1 | tee "${tmp_log}"

flash_count=$(grep -c "\[flash-decode\] online-softmax fastpath enabled" "${tmp_log}" || true)
if [[ ${flash_count} -ne 1 ]]; then
    echo "error: expected exactly one flash-decode banner, found ${flash_count}" >&2
    exit 3
fi

qgemv_count=$(grep -E "^\[qgemv\] (Q4_K|MXFP4) AVX2 decode kernel active \(n=1\)" "${tmp_log}" | wc -l | tr -d ' ')
if [[ ${qgemv_count} -ne 1 ]]; then
    echo "error: expected exactly one qGEMV banner, found ${qgemv_count}" >&2
    exit 4
fi

echo "Fast-path banners verified." >&2
