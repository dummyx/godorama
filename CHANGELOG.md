# Changelog

## [0.1.0] - 2026-04-01

### Added
- Initial project structure with CMake/Ninja build system
- CMakePresets.json with dev, release, asan, tsan presets
- llama.cpp integration via pinned submodule (commit `825eb91a6`)
- godot-cpp integration via pinned submodule (4.6-stable, tag `10.0.0-rc1`)
- `src/llama/`: Thin C++ wrapper layer around libllama
  - `LlamaModelHandle`: shared model ownership with tokenize/detokenize
  - `LlamaContextHandle`: move-only context with decode and embeddings
  - `LlamaSamplerHandle`: move-only sampler chain with configurable parameters
- `src/core/`: Async runtime
  - `InferenceWorker`: background jthread for non-blocking generation
  - Request queue with cooperative cancellation
  - Structured `Error` type with code/message/context
  - UTF-8 validation and codepoint counting
- `src/godot/`: GDExtension binding layer
  - `LlamaModelConfig` Resource with editor-friendly properties
  - `LlamaSession` RefCounted with async generation, tokenization, embeddings
  - Signal-based result delivery via `poll()`
- Unit tests (Catch2) for UTF-8, Error, and parameter types
- Minimal Godot demo project
- `.gdextension` manifest for Linux, Windows, macOS
- Documentation: BUILD.md, ARCHITECTURE.md, API.md
