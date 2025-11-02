# Benchmark Baseline (Step 0)

## Overview

This repo includes a reproducible baseline harness to measure decode TPS independently of prefill time.

Release builds are pinned via CMakePresets.json with OpenMP on and BLAS off.

## Build (choose one)

```bash
# Native CPU flags (recommended on the target machine)
cmake --preset release-native
cmake --build --preset build-release-native -j $(nproc)

# Or: portable build
cmake --preset release-portable
cmake --build --preset build-release-portable -j $(nproc)
```

## Run the baseline

```bash
# Run at 16 threads (adjust as needed)
tools/bench/run.sh 16
```

## What it does

- pins to NUMA node 0, binds threads to cores 0..THREADS-1
- sets OMP_NUM_THREADS=$THREADS, disables multi-threading in other BLAS libs
- runs perf stat with core cache / branch metrics
- executes: `./gptoss_main --threads $THREADS --prompt "<pangram>" --n-predict 512`

## TPS Definition

If the binary prints tokens/s (decode only), use that directly.

Otherwise compute:
`TPS = N_PRED / elapsed_decode_seconds` with `N_PRED=512`.

From perf output, parse the line containing: `X.XXXXX seconds time elapsed`.

## Example: compute TPS from perf (optional)

```bash
N_PRED=512
tools/bench/run.sh 16 2>perf.out || true
ELAPSED=$(awk '/seconds time elapsed/{print $1}' perf.out)
python - <<'PY'
n_pred = int("${N_PRED}")
elapsed = float("${ELAPSED}") if "${ELAPSED}" else float("nan")
print(f"Computed TPS (n_pred/elapsed): {n_pred/elapsed:.2f}" if elapsed==elapsed else "Could not parse elapsed time.")
PY
```

## Notes

Release builds define NDEBUG. If extra logging exists, disable via your runtime flag (e.g., --quiet) during benchmarks.

You can switch between release-native and release-portable without editing CMakeLists.txt.

## Troubleshooting

- Make sure numactl and perf are installed (`sudo apt-get install numactl linux-tools-common`).
- Ensure gptoss_main exists in the repo root or update the script path accordingly.
