# godorama

`godorama` is a Godot 4 GDExtension that embeds `llama.cpp` through `libllama` for local inference.

The repository now includes:

- async token generation through `LlamaSession`
- a local SQLite-backed RAG corpus with deterministic chunking and dense retrieval
- a grounded answer session with packed-context citations
- CMake + Ninja presets, unit tests, integration tests, and an offline evaluation harness

Build:

```sh
cmake --preset dev
cmake --build --preset build-dev
ctest --preset test-dev --output-on-failure
```

Key docs:

- `docs/BUILD.md`
- `docs/API.md`
- `docs/RAG_ARCHITECTURE.md`
- `docs/RAG_API.md`
- `docs/RAG_EVALUATION.md`
