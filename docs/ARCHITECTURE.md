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
│  LlamaModelConfig (Resource)        │
│  LlamaSession (RefCounted)          │
│  register_types.cpp                 │
└──────────────┬──────────────────────┘
               │ Internal C++ types
┌──────────────▼──────────────────────┐
│        src/core/                    │
│  InferenceWorker (jthread)          │
│  Request / RequestCallbacks         │
│  Error types                        │
│  UTF-8 helpers                      │
└──────────────┬──────────────────────┘
               │ llama.h API
┌──────────────▼──────────────────────┐
│        src/llama/                   │
│  LlamaModelHandle (shared_ptr)      │
│  LlamaContextHandle (move-only)     │
│  LlamaSamplerHandle (move-only)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  thirdparty/llama.cpp (libllama)    │
│  thirdparty/godot-cpp               │
└─────────────────────────────────────┘
```

## Layering rules

### src/llama/ (adapter layer)

- Wraps raw `llama_model`, `llama_context`, `llama_sampler`
- Translates `ModelConfig`/`GenerateOptions` to llama.cpp params
- No Godot headers, no Godot types
- Shields the rest of the codebase from upstream API churn

### src/core/ (runtime layer)

- `InferenceWorker`: background `jthread` that processes generation requests
- Request queue with cooperative cancellation
- `RequestCallbacks`: function-based interface for token/completion/error delivery
- UTF-8 validation and codepoint counting
- Structured `Error` type with code + message + context
- No Godot binding logic

### src/godot/ (binding layer)

- `LlamaModelConfig`: Godot `Resource` exposing model configuration as editor properties
- `LlamaSession`: Godot `RefCounted` with async generation, tokenization, embeddings
- Signal-based result delivery (emitted on main thread via `poll()`)
- All Godot ↔ C++ type conversions happen here

## Threading model

```
Main Thread                    Worker Thread
───────────                    ─────────────
session.open(config)
  ├── load model (blocking)
  └── start worker thread ──→  worker.run() loop
                                  wait on condition_variable
session.generate_async(prompt)
  └── enqueue request ───────→  dequeue request
                                  tokenize prompt
                                  decode prompt tokens
                                  sample loop:
                                    sample token
                                    on_token callback ───→ queued
                                  on_complete callback ──→ queued

session.poll() (each frame)
  ├── drain token_events → emit token_emitted signal
  ├── drain complete_events → emit completed signal
  ├── drain error_events → emit failed signal
  └── drain cancel_events → emit cancelled signal
```

## Ownership

- `LlamaModelHandle`: shared via `shared_ptr` (model can outlive individual contexts)
- `LlamaContextHandle`: move-only, holds a `shared_ptr` to its model
- `LlamaSamplerHandle`: move-only, created per-request, destroyed after request
- `InferenceWorker`: owns the context; owned by `LlamaSession`
- `GenerateRequest`: shared between queue and worker via `shared_ptr` for cancellation
