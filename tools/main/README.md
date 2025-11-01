# gptoss-cli

`gptoss-cli` is a minimal, fully-offline console runner for the GPT-OSS
runtime.  It links only against the first-party code shipped in this
repository and is intended for single-prompt CPU inference on Linux.

The program loads a GGUF model, evaluates the prompt, and streams the
model response to standard output using configurable sampling
parameters.  There is no interactive shell, no network access, and no
third-party dependencies.

## Basic usage

```bash
./gptoss-cli -m /path/to/model.gguf -p "Your prompt here"
```

The options recognised by the tool are:

- `-m, --model PATH` (required) – path to a local GGUF model file.
- `-p, --prompt TEXT` (default: BOS only) – prompt text to evaluate. If omitted the model is primed with just the BOS token.
- `-n, --n-predict N` (default: `-1`) – maximum tokens to sample. Leave at `-1` to stream until EOS; supply a positive integer to cap the response length.
- `-t, --threads N` (default: system thread count) – number of CPU threads used during generation.
- `-tb, --threads-batch N` (default: same as `--threads`) – threads dedicated to prompt ingestion.
- `--ctx-size N` (default: model setting) – override the context window (`n_ctx`) to reduce memory use or match saved KV caches.
- `--ubatch-size N` (default: model setting) – micro-batch size for prompt ingestion; also sets the evaluation batch size.
- `--mlock` – request `mlock` for model weights to reduce paging on Linux. Requires sufficient `ulimit -l`.
- `--numa MODE` (default: `none`) – NUMA policy (`none`, `distribute`, or `isolate`) for multi-socket systems.
- `--seed N` (default: `-1`) – seed for the internal PRNG; `-1` pulls from `std::random_device`.
- `--temp VALUE` (default: `0.8`) – sampling temperature. Lower values reduce randomness; `0` forces greedy decoding.
- `--top-p VALUE` (default: `0.95`) – nucleus sampling threshold; tokens are considered while cumulative probability stays below this value.
- `--top-k N` (default: `40`) – consider only the top-`k` logits before sampling (`0` disables the limit).
- `--repeat-penalty VALUE` (default: `1.1`) – penalty applied to recently generated tokens to discourage repetition.
- `--repeat-last-n N` (default: `64`) – number of most recent tokens tracked for the repetition penalty window.
- `--keep N` – accepted for compatibility but ignored. The CLI always feeds the entire prompt before generation.
- `--measure-tps` – run the 10 standard Developer Challenge prompts sequentially and print a True TPS table (no text output).

All options are entirely local—no network access or third-party services
are invoked.

## Measuring True TPS

Run the standardized throughput benchmark without tuning any inference
parameters:

```bash
./gptoss-cli -m models/gpt-oss-20b-Q4_K_M.gguf --measure-tps
```

The command reuses the current sampling configuration, evaluates the 10
prompts defined by the High-Performance Inference Engine challenge, and
prints a table that lists generated tokens, wall-clock time, and derived
TPS for each prompt. The footer row reports the overall True TPS using
the challenge formula (sum of generated tokens divided by the total
elapsed seconds). No files are written and no model outputs are printed
while the measurement runs.

## Example

Generate a short completion using all available CPU cores and a custom
seed:

```bash
./gptoss-cli \
  -m models/gpt-oss-20b-Q4_K_M.gguf \
  -p "Explain how GPT-OSS handles tokenizer merges." \
  -t $(nproc) \
  --seed 42 \
  --temp 0.7 \
  --top-p 0.9 \
  --top-k 100
```

The tokens will be written directly to standard output.  Use your
terminal's interrupt key (`Ctrl+C`) to stop generation early.
