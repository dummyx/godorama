# Build Instructions

## Prerequisites

- CMake 3.28+
- Ninja
- C++20 compiler (GCC 13+, Clang 16+, MSVC 2022+)
- Git (for submodules)

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

## GPU backends

CPU-only is the default. To enable GPU:

```sh
cmake --preset dev -DGGML_CUDA=ON    # NVIDIA CUDA
cmake --preset dev -DGGML_METAL=ON   # Apple Metal
cmake --preset dev -DGGML_VULKAN=ON  # Vulkan
```
