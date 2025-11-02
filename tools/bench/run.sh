#!/usr/bin/env bash
set -euo pipefail

PROMPT="The quick brown fox jumps over the lazy dog."
N_PRED=512
THREADS=${1:-16}
BIN="${BIN:-./build/bin/gptoss-cli}"
MODEL="${MODEL:-models/gpt-oss-20b-Q4_K_M.gguf}"

PHYS=$(tools/bench/cpu_pinning.sh)
NPHYS=$(( $(echo "$PHYS" | tr -cd ',' | wc -c) + 1 ))

decode_threads=${THREADS}
if [ "$THREADS" -gt "$NPHYS" ]; then
  decode_threads=$NPHYS
fi

echo "[bench] using physical cores: $PHYS (count=$NPHYS)  decode_threads=$decode_threads  prefill_threads=$THREADS"

is_perf_usable() {
  # perf exists AND does NOT print the infamous "not found for kernel" warning
  command -v perf >/dev/null 2>&1 || return 1
  ! perf --version 2>&1 | grep -q 'not found for kernel'
}

inner_script() {
  cat <<EOS
$(declare -f is_perf_usable)
  if is_perf_usable; then
    perf stat -e cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,cache-misses \
      "$BIN" -m "$MODEL" -p "$PROMPT" -n $N_PRED -t $decode_threads -tb $THREADS --numa none
  else
    if command -v perf >/dev/null 2>&1; then
      echo "[bench] perf present but unusable for this kernel -> skipping HW counters"
    fi
    "$BIN" -m "$MODEL" -p "$PROMPT" -n $N_PRED -t $decode_threads -tb $THREADS --numa none
  fi
EOS
}

# NUMA pin + CPU pin + openmp threads fixed for noisy libs
numactl --cpunodebind=0 --membind=0 \
  taskset -c "$PHYS" \
  env \
    OMP_NUM_THREADS=$decode_threads \
    OMP_PROC_BIND=TRUE \
    OMP_PLACES=threads \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
  bash -lc "$(inner_script)"
