# Build Instructions

## Prerequisites

- CMake 3.28+
- Ninja
- C++20 compiler (GCC 13+, Clang 16+, MSVC 2022+)
- Git (for submodules)
- A checked-out `thirdparty/libsql` tree at commit `0653c5788d77ef16a97c56ff3e9fdc11717a72d9`
  (the build uses the vendored libSQL amalgamation under `libsql-ffi/bundled/src/`)

### Optional

- ccache (Linux/macOS) or sccache (Windows) for build caching
- clang-format and clang-tidy for code quality targets

## First-time setup

```sh
git submodule update --init --recursive
git clone https://github.com/tursodatabase/libsql.git thirdparty/libsql
git -C thirdparty/libsql checkout 0653c5788d77ef16a97c56ff3e9fdc11717a72d9
```

## Build commands

### Development (debug, with tests)

```sh
cmake --preset dev
cmake --build --preset build-dev
ctest --preset test-dev --output-on-failure
```

To disable the multimodal scaffold entirely:

```sh
cmake --preset dev -DGODOT_LLAMA_ENABLE_MTMD=OFF
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
- `GODOT_LLAMA_ENABLE_MTMD=ON` keeps the `libmtmd` multimodal scaffold available; the current Godot surface exposes projector loading and capability introspection, not media generation requests yet.

## GPU backends

CPU-only is the default. To enable GPU:

```sh
cmake --preset dev -DGGML_CUDA=ON    # NVIDIA CUDA
cmake --preset dev -DGGML_METAL=ON   # Apple Metal
cmake --preset dev -DGGML_VULKAN=ON  # Vulkan
```
