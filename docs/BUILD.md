# Build Instructions

## Prerequisites

- CMake 3.28+
- Ninja
- C++20 compiler (GCC 13+, Clang 16+, MSVC 2022+)
- Git (for submodules)
- Tcl shell and `make`/`nmake` are required to generate the bundled SQLite amalgamation during the build

### Optional

- ccache (Linux/macOS) or sccache (Windows) for build caching
- clang-format and clang-tidy for code quality targets

## First-time setup

```sh
git submodule update --init --recursive
```

## Build commands

### Development (debug, with tests)

```sh
cmake --preset dev
cmake --build --preset build-dev
ctest --preset test-dev --output-on-failure
```

If a sandboxed launcher interferes with `ctest`, run the binaries directly:

```sh
./build/dev/tests/unit/godot_llama_tests
./build/dev/tests/integration/godot_llama_rag_integration_tests
./build/dev/tests/integration/godot_llama_rag_eval
```

### RAG evaluation

```sh
./build/dev/tests/integration/godot_llama_rag_eval
```

### Release

```sh
cmake --preset release
cmake --build --preset build-release
```

### Sanitizers

AddressSanitizer + UBSan:

```sh
cmake --preset asan
cmake --build --preset build-asan
```

ThreadSanitizer:

```sh
cmake --preset tsan
cmake --build --preset build-tsan
```

### Format and lint

```sh
cmake --build --preset build-dev --target format        # auto-format
cmake --build --preset build-dev --target format-check  # check only
cmake --build --preset build-dev --target lint           # clang-tidy
```

## Build outputs

- GDExtension library: `demo/bin/libgodot_llama.<platform>.<target>.<arch>.so|dll|dylib`
- compile_commands.json: `build/dev/compile_commands.json`
- Unit test binary: `build/dev/tests/unit/godot_llama_tests`
- Integration test binary: `build/dev/tests/integration/godot_llama_rag_integration_tests`
- Evaluation binary: `build/dev/tests/integration/godot_llama_rag_eval`

## Runtime Integration Notes

- `LlamaSession` and `LlamaEvalSession` both require `poll()` to flush queued signals on the Godot main thread.
- `LlamaSession.open()` and `LlamaEvalSession.open()` are asynchronous.
- Runtime log verbosity can be overridden with `GODORAMA_LLAMA_LOG_LEVEL=debug|info|warn|error|silent`.
- Message-based chat generation uses the `llama.cpp` common-chat Jinja engine, not the limited `llama_chat_apply_template()` helper.

## GPU backends

CPU-only is the default. To enable GPU:

```sh
cmake --preset dev -DGGML_CUDA=ON    # NVIDIA CUDA
cmake --preset dev -DGGML_METAL=ON   # Apple Metal
cmake --preset dev -DGGML_VULKAN=ON  # Vulkan
```
