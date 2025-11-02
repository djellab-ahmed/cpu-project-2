#!/usr/bin/env bash
set -euo pipefail
PROMPT="The quick brown fox jumps over the lazy dog."
N_PRED=512
THREADS=${1:-16}
MODE="${MODE:-ht}"

if [[ "$MODE" == "phys" ]]; then
  MASK="$(tools/bench/cpu_pinning.sh)"
  DECODE_THREADS="$(awk -F, '{print NF}' <<<"$MASK")"
else
  MASK="0-$((THREADS-1))"
  DECODE_THREADS="$THREADS"
fi

CMD="./build/bin/gptoss-cli -m models/gpt-oss-20b-Q4_K_M.gguf -p \"$PROMPT\" -n $N_PRED -t $DECODE_THREADS -tb $THREADS --ubatch-size 1024 --numa none --ctx-size 8192"

numactl --cpunodebind=0 --membind=0 \
taskset -c "$MASK" \
bash -lc '
  env \
    OMP_NUM_THREADS='"'"'$DECODE_THREADS'"'"' \
    OMP_DYNAMIC='"'"'${OMP_DYNAMIC:-TRUE}'"'"' \
    OMP_PROC_BIND=close \
    OMP_PLACES=cores \
    OMP_WAIT_POLICY=PASSIVE \
    GOMP_CPU_AFFINITY="'"'"$MASK"'"'" \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
  '"'"$CMD"'"''
'
