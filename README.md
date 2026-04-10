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
  - multimodal generation with image/audio inputs
  - tokenization / detokenization
  - text embeddings
  - LoRA adapter count and multimodal capability introspection
  - request-scoped multimodal token accounting
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
- Multimodal requests are submitted through `generate_multimodal_async()` / `generate_multimodal_messages_async()`. The Godot layer validates media dictionaries, file readability, and media-marker counts before the request reaches the worker thread.
- Completed multimodal requests include `multimodal_token_count` in their `completed` stats, and `get_multimodal_token_count(request_id)` returns the stored count for completed multimodal requests.
- RAG storage is local-only and embedded. `RagCorpus` uses libSQL in-process, with exact cosine retrieval executed in SQL by default.
- The current RAG surface is intentionally cosine-only. The old `vector_metric` setting is no longer part of `RagCorpusConfig`.
- Runtime log verbosity is filtered by default. Set `GODORAMA_LLAMA_LOG_LEVEL=debug|info|warn|error|silent` to override.

## Build

```sh
cmake --preset dev
cmake --build --preset build-dev
ctest --preset test-dev --output-on-failure
```

The current build also expects a `thirdparty/libsql` checkout at commit
`0653c5788d77ef16a97c56ff3e9fdc11717a72d9`. `docs/BUILD.md` includes the exact
setup commands.

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
