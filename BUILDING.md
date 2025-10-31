# Building and Running GPTOSS on Linux

The project has been renamed from the original llama.cpp sources, so any cached
CMake state or compiled binaries that still reference the old target names need
to be discarded before rebuilding. The safest path is to remove the previous
`build/` directory and then configure a fresh Release build.

## Prerequisites

Model downloads are now handled exclusively through libcurl, so make sure the
development headers for curl are present on your system before configuring the
project. On Debian/Ubuntu the package is called `libcurl4-openssl-dev`:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libcurl4-openssl-dev
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
options that keep only the command-line tools:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGPTOSS_BUILD_TESTS=OFF -DGPTOSS_BUILD_EXAMPLES=OFF \
  -DGPTOSS_BUILD_SERVER=OFF -DGPTOSS_BUILD_TOOLS=ON
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
  --keep 0 \
  --prompt "Explain how GPT-OSS handles tokenizer merges."
```

These steps mirror the previous guidance but ensure you start from a clean
build so that all GPTOSS identifiers are picked up correctly.
