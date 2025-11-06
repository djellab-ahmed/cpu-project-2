# Phase-0 TPS Benchmarking

Use this checklist every time you want to confirm throughput after a change. It keeps the build, benchmark suite, and diagnostics consistent with the Phase-0 instrumentation.

1. **Build (release)**
   ```bash
   cmake --preset release-native
   cmake --build --preset build-release-native -j"$(nproc)"
   ```

2. **(Optional) One warm-up run**
   Keeps caches/JITs/filesystem warm so your three measured runs are stable.
   ```bash
   ./build/bin/gptoss-cli -m models/gpt-oss-20b-Q4_K_M.gguf -t 16 -n 64 --prompt "warmup" >/dev/null
   ```

3. **Run the Phase-0 suite (3× and average)**
   Use the same 10 prompts, same model, same threads, same n-predict as your baseline.
   ```bash
   # Run 1
   ./tools/bench/baseline.sh \
     models/gpt-oss-20b-Q4_K_M.gguf 16 256 \
     tools/bench/prompts10.txt tools/bench/logs/suite_1.json

   # Run 2
   ./tools/bench/baseline.sh \
     models/gpt-oss-20b-Q4_K_M.gguf 16 256 \
     tools/bench/prompts10.txt tools/bench/logs/suite_2.json

   # Run 3
   ./tools/bench/baseline.sh \
     models/gpt-oss-20b-Q4_K_M.gguf 16 256 \
     tools/bench/prompts10.txt tools/bench/logs/suite_3.json

   # Quickly read the True TPS
   jq -r '.suite.true_tps' tools/bench/logs/suite_1.json
   jq -r '.suite.true_tps' tools/bench/logs/suite_2.json
   jq -r '.suite.true_tps' tools/bench/logs/suite_3.json

   # Compute the mean and stdev (helper)
   python3 - <<'PY'
   import json, sys, statistics as st, glob
   vals=[json.load(open(p))["suite"]["true_tps"] for p in glob.glob("tools/bench/logs/suite_*.json")]
   print("runs:", vals, "\nmean:", sum(vals)/len(vals), "stdev:", st.pstdev(vals))
   PY
   ```
   Acceptance check: mean stable within ±1% across the three runs.

4. **Capture per-token traces (diagnostics)**
   Enable JSON node timing per run (already wired by Phase 0).
   ```bash
   GGML_SCHED_DEBUG=2 \
   ./build/bin/gptoss-cli \
     -m models/gpt-oss-20b-Q4_K_M.gguf \
     -t 16 -n 256 \
     --prompt-file tools/bench/prompts10.txt \
     --bench-json /tmp/bench_trace_after.json

   jq '.nodes | group_by(.name) | map({name: .[0].name, total_us: (map(.dur_us)|add)}) | sort_by(.total_us) | reverse[:20]' \
     /tmp/bench_trace_after.json
   ```

5. **Cross-check with perf (one wrapped suite run)**
   Confirms CPU-side changes (LLC/TLB/branch).
   ```bash
   perf stat -e cycles,instructions,branch-misses,L1-dcache-load-misses,LLC-load-misses \
     ./tools/bench/baseline.sh \
       models/gpt-oss-20b-Q4_K_M.gguf 16 256 \
       tools/bench/prompts10.txt tools/bench/logs/suite_perf.json
   ```

6. **A/B compare vs baseline**
   ```bash
   jq -r '.suite.true_tps' tools/bench/logs/suite_before.json
   jq -r '.suite.true_tps' tools/bench/logs/suite_after.json

   paste <(jq -r '.nodes|group_by(.name)|map({k:.[0].name,v:(map(.dur_us)|add)})|from_entries|to_entries|sort_by(.key)|.[]|"\(.key)\t\(.value)"' /tmp/bench_trace_before.json) \
         <(jq -r '.nodes|group_by(.name)|map({k:.[0].name,v:(map(.dur_us)|add)})|from_entries|to_entries|sort_by(.key)|.[]|"\(.key)\t\(.value)"' /tmp/bench_trace_after.json) \
   | awk -F'\t' 'BEGIN{printf "%-32s %12s %12s %8s\n","node","before_us","after_us","delta%"} {if(NF==4){b=$2;a=$4;d=(b-a)/b*100;printf "%-32s %12d %12d %7.1f%%\n",$1,b,a,d}}'
   ```

7. **One-liner for quick TPS check**
   ```bash
   ./tools/bench/baseline.sh models/gpt-oss-20b-Q4_K_M.gguf 16 256 tools/bench/prompts10.txt /tmp/suite.json >/dev/null && \
   jq -r '.suite.true_tps' /tmp/suite.json
   ```

## Tips for clean comparisons

- Fix temperature=0.0 and the same seed if your CLI exposes them, so token paths are identical.
- Keep prompts, n_predict, threads constant between runs.
- Run on an idle machine; disable CPU frequency scaling if possible (governor performance).
- If variance >1.5%, increase `n_predict` (e.g., 512) so decode dominates.

With this checklist you have repeatable TPS numbers, per-token traces, and perf counters ready to spot regressions after each optimization.
