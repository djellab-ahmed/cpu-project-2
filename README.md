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

## Quick start

### Build (native Release)
```bash
cmake --preset release-native
cmake --build --preset build-release-native -j"$(nproc)"

Download model (HF CLI)
./scripts/hf_download.sh
# Outputs: models/gpt-oss-20b-Q4_K_M.gguf

Run a quick bench
./tools/bench/run.sh 16

Reproducible TPS baseline + log
./tools/bench/baseline.sh models/gpt-oss-20b-Q4_K_M.gguf 16 1024 8192
# Logs are saved under tools/bench/logs/

Note on --mlock

If you see an mlock warning, either drop --mlock or raise limits:

sudo sh -c 'echo "* soft memlock unlimited" >> /etc/security/limits.conf'
sudo sh -c 'echo "* hard memlock unlimited" >> /etc/security/limits.conf'
ulimit -l unlimited


---

### Acceptance checks (run before committing)
1) `bash -n tools/bench/run.sh && bash -n tools/bench/baseline.sh` passes.  
2) `cmake --preset release-native && cmake --build --preset build-release-native -j"$(nproc)"` succeeds and produces `./build/bin/gptoss-cli`.  
3) `./scripts/hf_download.sh` creates `models/gpt-oss-20b-Q4_K_M.gguf` (no commit of model).  
4) `./tools/bench/run.sh 2` executes without errors (skip perf counters automatically if unsupported).  
5) `.gitignore` excludes `build/`, `models/*` (except README/.gitkeep), logs, and typical local artifacts.

---

### Commit & PR
- Commit in logical units with messages:
  - `bench: standardize run.sh on gptoss-cli with perf gating`
  - `bench: add baseline.sh with logging under tools/bench/logs`
  - `build: add CMakePresets.json (release-native)`
  - `repo: .gitignore for builds/models/logs/env`
  - `scripts: add hf_download.sh`
  - `docs: update README quick start (build/bench/hf/memlock)`
- Open a PR titled: **“Bench cleanup: standardize on gptoss-cli, presets, .gitignore, HF helper”**  
- PR description: summarize the 4 objectives and list the acceptance checks.
