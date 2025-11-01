# Building and Running GPTOSS on Linux

The project has been renamed from the original llama.cpp sources, so any cached
CMake state or compiled binaries that still reference the old target names need
to be discarded before rebuilding. The safest path is to remove the previous
`build/` directory and then configure a fresh Release build.

## Prerequisites

Only a standard C/C++ toolchain and CMake are required. On Debian/Ubuntu you
can install the necessary packages with:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
```

## 1. Clean up previous build output

If you have an older build tree (for example one that still produced
`llama-cli`), delete it to avoid picking up stale cache entries:

```bash
rm -rf build/
```

Alternatively, from inside the repository you can invoke the CMake clean target
against the existing directory:

```bash
cmake --build build --target clean
```

Removing the directory is preferred when switching between different versions
of the project because it guarantees that CMake will regenerate the cache using
only the current GPTOSS options.

## 2. Configure a Release build

Create a new build directory and run CMake with the GPTOSS-specific cache
option that enables just the command-line tools:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGPTOSS_BUILD_TOOLS=ON
```

The output binaries are written to `build/bin/`.

## 3. Compile the CLI

Build the renamed executable:

```bash
cmake --build build --target gptoss-cli
```

## 4. Run inference

Once the model file (for example, `models/gpt-oss-20b-Q4_K_M.gguf`) is in
place, launch the CLI:

```bash
./build/bin/gptoss-cli \
  -m models/gpt-oss-20b-Q4_K_M.gguf \
  -t $(nproc) -tb $(nproc) \
  --ubatch-size 1024 \
  --mlock \
  --numa distribute \
  --ctx-size 8192 \
  --prompt "Explain how GPT-OSS handles tokenizer merges."
```

Additional sampling controls such as `--temp`, `--top-p`, `--top-k`,
`--repeat-penalty`, and `--repeat-last-n` are available if you need to
tune the diversity of the generated output.  Passing `--seed` allows
you to reproduce a particular run.

These steps mirror the previous guidance but ensure you start from a clean
build so that all GPTOSS identifiers are picked up correctly.
