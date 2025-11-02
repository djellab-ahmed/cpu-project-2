# Benchmarking checklist

Use this checklist to keep throughput measurements consistent and reproducible:

- ✅ One pool (OpenMP only); BLAS forced to 1 if enabled.
- ✅ Binding: `taskset` mask + `OMP_PLACES=cores` + `OMP_PROC_BIND=close`.
- ✅ NUMA policy consistent: `--numa none` + `--membind 0` or `--numa interleave` (not both).
- ✅ Decode threads: sweep 12/14/16; keep winner.
- ✅ Prefill threads: `-tb 16` unless it hurts early decode.
- ✅ `mlock`: raise limit or remove `--mlock`.
- ✅ Release flags: `-O3 -DNDEBUG -march=native` (+ `-fno-math-errno` / `-fno-trapping-math` for ggml).
- ✅ Warm up once before measuring.

Warm-up example:

```bash
./build/bin/gptoss-cli -m models/gpt-oss-20b-Q4_K_M.gguf \
  -p warmup -n 64 -t 16 -tb 16 --ubatch-size 1024 --numa none >/dev/null 2>&1
```

Thread sweep example:

```bash
for T in 12 14 16; do
  MODE=ht ./tools/bench/baseline.sh models/gpt-oss-20b-Q4_K_M.gguf "$T" 1024 8192 | tee -a sweep.log
done
```
