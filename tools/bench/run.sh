#!/usr/bin/env bash
set -euo pipefail
PROMPT="The quick brown fox jumps over the lazy dog."
N_PRED=512
THREADS=${1:-16}

# single NUMA node + explicit core pinning
numactl --cpunodebind=0 --membind=0 \
taskset -c 0-$((THREADS-1)) \
env OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
perf stat -e cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,cache-misses \
./gptoss_main --threads $THREADS --prompt "$PROMPT" --n-predict $N_PRED
