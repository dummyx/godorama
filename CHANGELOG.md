# Changelog

## [Unreleased] - 2026-04-03

### Added
- `LlamaLoraAdapterConfig` resource and `LlamaModelConfig.lora_adapters`
- `LlamaMultimodalConfig` resource and `LlamaModelConfig.multimodal_config`
- LoRA adapter loading through the stable `llama.cpp` adapter API during session open
- optional `libmtmd` build wiring behind `GODOT_LLAMA_ENABLE_MTMD`
- `LlamaSession` capability introspection for loaded LoRA adapter count plus image/audio multimodal support
- `LlamaSession.generate_multimodal_async()` for multimodal generation with image/audio inputs
- `LlamaSession.generate_multimodal_messages_async()` for message-templated multimodal generation
- in-memory `PackedByteArray` media input support via `data` key in media input dictionaries
- `MultimodalInput.data` field for passing image/audio bytes without a file path

### Changed
- Multimodal generation methods are now fully bound and documented
- `LlamaMultimodalHandle::evaluate_prompt` accepts both file-path and in-memory buffer media
- Media input dictionaries now accept either `path` or `data` (or both, with `data` taking precedence)
- Updated `API.md` with full multimodal method documentation and media input dictionary format

## [0.2.0] - 2026-04-02

### Added
- first-class local RAG support
- `RagCorpusConfig`, `RagCorpus`, and `RagAnswerSession` Godot classes
- SQLite-backed persistent corpus storage with schema versioning
- deterministic UTF-8 normalization and token-aware chunking
- dedicated embedding pipeline through `libllama`
- exact dense retrieval with metadata/source filtering, dedupe, and MMR
- grounded prompt assembly with packed-chunk citations
- mock-embedder-backed RAG unit tests and SQLite integration tests
- offline evaluation executable and RAG documentation set
- updated demo project with local corpus ingestion, retrieval preview, and streamed grounded answers

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
