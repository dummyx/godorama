# Architecture

## Layer diagram

```
┌─────────────────────────────────────┐
│         Godot Scene Tree            │
│  (GDScript / C# / Visual Script)   │
└──────────────┬──────────────────────┘
               │ Variant-compatible API
┌──────────────▼──────────────────────┐
│        src/godot/                   │
│  LlamaLoraAdapterConfig (Resource)  │
│  LlamaMultimodalConfig (Resource)   │
│  LlamaModelConfig (Resource)        │
│  LlamaSession (RefCounted)          │
│  LlamaEvalSession (RefCounted)      │
│  RagCorpusConfig (Resource)         │
│  RagCorpus (RefCounted)             │
│  RagAnswerSession (RefCounted)      │
│  register_types.cpp                 │
└──────────────┬──────────────────────┘
               │ Internal C++ types
┌──────────────▼──────────────────────┐
│        src/core/                    │
│  InferenceWorker (jthread)          │
│  request queues / cancellation      │
│  Request / RequestCallbacks         │
│  Error types                        │
│  UTF-8 helpers                      │
│  rag::CorpusEngine                  │
│  rag::Chunker / Embedder            │
│  rag::CorpusStore / Retriever       │
│  rag::ContextPacker / Reranker      │
└──────────────┬──────────────────────┘
               │ llama.h API
┌──────────────▼──────────────────────┐
│        src/llama/                   │
│  LlamaModelHandle (shared_ptr)      │
│  LlamaContextHandle (move-only)     │
│  LlamaLoraAdapterHandle (move-only) │
│  LlamaMultimodalHandle (move-only)  │
│  LlamaSamplerHandle (move-only)     │
│  ChatTemplateEngine                 │
│  position layout helpers            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  thirdparty/llama.cpp (libllama)    │
│  thirdparty/llama.cpp/tools/mtmd    │
│  thirdparty/godot-cpp               │
└─────────────────────────────────────┘
```

## Layering rules

### src/llama/ (adapter layer)

- Wraps raw `llama_model`, `llama_context`, `llama_sampler`
- Owns LoRA adapter lifetime against the loaded `llama_model`
- Optionally owns `libmtmd` projector state for multimodal capability probing
- Translates `ModelConfig`/`GenerateOptions` to llama.cpp params
- Owns the reusable chat-template engine over `llama.cpp/common/chat.*`
- Normalizes multi-component position layouts for eval-style embedding prefill
- No Godot headers, no Godot types
- Shields the rest of the codebase from upstream API churn

### src/core/ (runtime layer)

- `InferenceWorker`: background `jthread` that processes generation requests
- Request queue with cooperative cancellation
- `RequestCallbacks`: function-based interface for token/completion/error delivery
- UTF-8 validation and codepoint counting
- Structured `Error` type with code + message + context
- Message templating is delegated to the model/chat-template layer before submission
- `rag::CorpusEngine`: synchronous core RAG coordinator
- `rag::LibSqlCorpusStore`: schema-versioned persistent corpus store backed by embedded libSQL vector SQL
- `rag::DeterministicChunker`: token-aware chunking with stable offsets/IDs
- `rag::LlamaEmbedder`: dedicated embedding model/context path
- `rag::DenseRetriever`: exact cosine SQL retrieval with dedupe and MMR
- `rag::GroundedContextPacker`: token-budgeted grounded prompt assembly
- No Godot binding logic

### src/godot/ (binding layer)

- `LlamaModelConfig`: Godot `Resource` exposing model configuration as editor properties
- `LlamaLoraAdapterConfig`: Godot `Resource` for adapter path + scale
- `LlamaMultimodalConfig`: Godot `Resource` for `mmproj` configuration
- `LlamaSession`: Godot `RefCounted` with async generation, tokenization, embeddings
- `LlamaEvalSession`: Godot `RefCounted` with async `inputs_embeds` prefill/eval
- `RagCorpusConfig`: editor-facing corpus and embedding configuration
- `RagCorpus`: background ingestion/retrieval queue with main-thread signal delivery
- `RagAnswerSession`: retrieval + prompt assembly + streaming generation orchestration
- Signal-based result delivery (emitted on main thread via `poll()`)
- All Godot ↔ C++ type conversions happen here

## Threading model

```
Main Thread                    Background Threads
───────────                    ──────────────────
session.open(config)
  ├── start async open thread ─→ load model
  │                             load LoRA adapters
  │                             create text context
  │                             optionally create mtmd projector
  └── return immediately

session.poll()
  └── emits opened() / failed()

session worker startup ───────→ worker.run() loop
                                  wait on condition_variable
session.generate_async(prompt)
  └── enqueue request ───────→  dequeue request
                                  tokenize prompt
                                  decode prompt tokens
                                  sample loop:
                                    sample token
                                    on_token callback ───→ queued
                                  on_complete callback ──→ queued
session.generate_multimodal_async(prompt, media_inputs)
  ├── validate media dictionaries, readable paths, marker count
  └── enqueue request ───────→  dequeue request
                                  mtmd tokenize + media encode
                                  decode multimodal prompt
                                  shared sampling loop
                                  on_complete callback ──→ queued

session.poll() (each frame)
  ├── drain token_events → emit token_emitted signal
  ├── drain complete_events → emit completed signal
  ├── drain error_events → emit failed signal
  └── drain cancel_events → emit cancelled signal
```

`LlamaEvalSession` follows the same high-level pattern, but its worker queue accepts embedding-prefill requests instead of text generation requests.

## Chat Templating

- `LlamaSession.generate_messages_async()` converts Godot `{role, content}` dictionaries to internal message pairs.
- The prompt is rendered with `ChatTemplateEngine`.
- `ChatTemplateEngine` uses `llama.cpp` `common/chat.*` with `use_jinja = true`.
- `disable_thinking` is passed through `enable_thinking = !disable_thinking`.

This matters because the older `llama_chat_apply_template()` path is intentionally not the templating engine used here.

## Eval Session Contract

`LlamaEvalSession` is designed for callers that already have embedding inputs.

Current result contract:

- `logits: PackedFloat32Array`
- `logits_shape: PackedInt64Array`
- `hidden_states: PackedFloat32Array` when requested
- `hidden_states_shape: PackedInt64Array` when requested

The current implementation returns the final-step logit slice and, when requested, the final hidden-state row.

## Logging

`register_types.cpp` installs a process-wide `llama.cpp` log callback:

- default threshold: `warn`
- override with `GODORAMA_LLAMA_LOG_LEVEL`

This keeps downstream Godot runs readable while still surfacing real warnings and errors.

## Ownership

- `LlamaModelHandle`: shared via `shared_ptr` (model can outlive individual contexts)
- `LlamaContextHandle`: move-only, holds a `shared_ptr` to its model
- `LlamaLoraAdapterHandle`: move-only, owned by `LlamaModelHandle`, freed before the model handle
- `LlamaMultimodalHandle`: move-only, owned by `InferenceWorker`, freed before the model handle
- `LlamaSamplerHandle`: move-only, created per-request, destroyed after request
- `InferenceWorker`: owns the context; owned by `LlamaSession`
- `LlamaEvalSession`: owns its own context, worker thread, and eval request queue
- `GenerateRequest`: shared between queue and worker via `shared_ptr` for cancellation
- `rag::CorpusEngine`: owns store/chunker/embedder/retriever/reranker behind a shared core handle
- `RagCorpus`: owns a serial background job queue for ingest/retrieve operations
- `RagAnswerSession`: owns a separate answer queue plus an `InferenceWorker` for streaming generation

## Multimodal Status

- LoRA support uses the stable `llama.h` adapter API and is wired as a usable scaffold today.
- Image/audio support is intentionally split behind `LlamaMultimodalHandle` and `LlamaMultimodalConfig`.
- The Godot-facing surface includes `generate_multimodal_async()` and `generate_multimodal_messages_async()` with file-path or in-memory media dictionaries.
- `LlamaSession` performs synchronous media dictionary validation in the Godot layer, then the worker uses `LlamaMultimodalHandle::evaluate_prompt()` to tokenize media and decode the combined prompt off the main thread.
- Multimodal accounting stays request-scoped: completed multimodal requests carry `multimodal_token_count` in `completed` stats and can be queried later by `request_id`.
