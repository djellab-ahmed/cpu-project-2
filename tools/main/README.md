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

| Option | Description |
| --- | --- |
| `-m, --model PATH` | **Required.** Path to a local GGUF model file. |
| `-p, --prompt TEXT` | Prompt text to evaluate. If omitted the model will be primed with only the BOS token. |
| `-n, --n-predict N` | Maximum number of tokens to generate (default unlimited until EOS; provide a positive value to cap output). |
| `-t, --threads N` | Number of threads used during generation. |
| `-tb, --threads-batch N` | Number of threads used when ingesting the prompt. |
| `--ctx-size N` | Override the context window (`n_ctx`). |
| `--ubatch-size N` | Micro-batch size for prompt ingestion; also sets the batch size. |
| `--mlock` | Request `mlock` for model weights to reduce paging. |
| `--numa MODE` | NUMA policy: `none`, `distribute`, or `isolate`. |
| `--seed N` | PRNG seed (`-1` uses `std::random_device`). |
| `--temp VALUE` | Sampling temperature (default `0.8`). Set to `0` for greedy decoding. |
| `--top-p VALUE` | Nucleus sampling threshold (default `0.95`). |
| `--top-k N` | Consider only the top-`k` logits before sampling (default `40`, `0` disables the limit). |
| `--repeat-penalty VALUE` | Penalty applied to recently generated tokens (default `1.1`). |
| `--repeat-last-n N` | Size of the history window for the repetition penalty (default `64`). |

For compatibility with historical scripts the `--keep` flag is
accepted but ignored; the CLI always consumes the entire prompt before
generation.

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
