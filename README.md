# godorama

`godorama` is a Godot 4 GDExtension that embeds `llama.cpp` directly through `libllama`.

It is intended to be a production-facing local runtime for Godot projects, not a subprocess wrapper.

## Current Public Surface

- `LlamaLoraAdapterConfig`
  - editor-friendly `Resource` for LoRA adapter path + scale
- `LlamaMultimodalConfig`
  - editor-friendly `Resource` for `mmproj` projector configuration
- `LlamaModelConfig`
  - editor-friendly `Resource` for GGUF model configuration
  - optional LoRA adapter array
  - optional multimodal projector config
- `LlamaSession`
  - async text generation
  - message-based chat templating
  - tokenization / detokenization
  - text embeddings
  - LoRA adapter count and multimodal capability introspection
- `LlamaEvalSession`
  - async embedding-prefill evaluation over `inputs_embeds`
  - optional hidden-state readback
  - intended for native evaluation workflows such as hybrid speech runtimes
- `RagCorpusConfig`
- `RagCorpus`
- `RagAnswerSession`

## Important Implementation Notes

- Model open is asynchronous. Godot callers must use `poll()` to flush queued signals on the main thread.
- LoRA adapters are loaded at `open()` time through `llama.cpp`'s stable adapter API and applied when the session context is created.
- Chat message templating goes through `llama.cpp`'s `common/chat.*` Jinja path, not the limited `llama_chat_apply_template()` helper.
- `disable_thinking` is meaningful for message-templated generation. It is not a magic prompt rewrite for raw `generate_async(prompt)`.
- `LlamaEvalSession` is the supported way to run prefill/eval style embedding inputs from Godot without exposing raw `llama_*` handles.
- Multimodal configuration currently scaffolds `libmtmd` loading and image/audio capability detection. It does not yet expose a public Godot media-generation request API.
- Runtime log verbosity is filtered by default. Set `GODORAMA_LLAMA_LOG_LEVEL=debug|info|warn|error|silent` to override.

## Build

```sh
cmake --preset dev
cmake --build --preset build-dev
ctest --preset test-dev --output-on-failure
```

Optional multimodal scaffold:

```sh
cmake --preset dev -DGODOT_LLAMA_ENABLE_MTMD=ON
```

If a sandboxed environment interferes with the `ctest` wrapper itself, run the test binaries directly from `build/dev/tests/...` to distinguish a harness problem from a test failure.

## Key Docs

- `docs/BUILD.md`
- `docs/API.md`
- `docs/ARCHITECTURE.md`
- `docs/RAG_ARCHITECTURE.md`
- `docs/RAG_API.md`
- `docs/RAG_EVALUATION.md`
