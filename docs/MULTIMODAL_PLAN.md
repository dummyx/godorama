# Multimodal Implementation Plan

## Current State

The core multimodal pipeline is already functional:

- **Wrapper layer**: `LlamaMultimodalHandle` wraps `libmtmd` with bitmap loading,
  tokenization, chunk evaluation, and capability queries (`src/llama/llama_multimodal_handle.cpp`).
- **Config**: `LlamaMultimodalConfig` Godot `Resource` with 7 properties
  (`mmproj_path`, `media_marker`, `use_gpu`, `print_timings`, `n_threads`,
  `image_min_tokens`, `image_max_tokens`).
- **Godot API**: `generate_multimodal_async()` and
  `generate_multimodal_messages_async()` on `LlamaSession`, plus
  `supports_image_input()`, `supports_audio_input()`,
  `get_audio_input_sample_rate_hz()`.
- **Worker**: `InferenceWorker` handles multimodal request submission, media
  evaluation in `process_request`, and token sampling after multimodal prefill.
- **Input types**: `MultimodalInput` supports file paths and in-memory
  `PackedByteArray` buffers, with `Image` and `Audio` type tags.
- **Build**: `GODOT_LLAMA_ENABLE_MTMD` cmake option (default ON), conditional
  compilation throughout.
- **Unit tests**: Handle construction, move semantics, error paths
  (`tests/unit/test_multimodal_handle.cpp`).
- **Docs**: Full method signatures and media input dictionary format in
  `docs/API.md`.

The multimodal API is bound, documented, and shipped in the `[Unreleased]`
changelog. What remains is hardening, test coverage, a demo, and selective
exposure of upstream capabilities.

---

## Cross-Cutting Constraints

These constraints apply to every phase in this plan:

- Preserve the existing async Godot contract: `open()` remains asynchronous,
  request submission remains non-blocking, and all user-visible signals are
  still delivered through `poll()`.
- Keep the public surface media-generic. Do not hard-code image-only naming or
  assumptions into core request, progress, caching, or accounting APIs unless a
  feature is intentionally image-specific.
- Keep request-scoped state request-scoped. Progress, token accounting, cache
  hits, and future metadata must be attributable to a specific `request_id`
  rather than exposed as ambiguous session-global "last" state.
- Preserve stable media identifiers (`id`) and media provenance in internal
  types so future multimodal RAG, richer audio pipelines, and eventual video
  support are not blocked by image-centric shortcuts taken now.
- Keep media preprocessing isolated behind the `src/llama/` and `src/core/`
  seams rather than baking format-specific behavior into the Godot binding
  layer.

---

## Phase 0 -- Documentation Sync

**Goal**: Eliminate stale claims that contradict the shipped API.

### 0.1 Fix README.md

`README.md:38` says:

> Multimodal configuration currently scaffolds libmtmd loading and image/audio
> capability detection. It does not yet expose a public Godot
> media-generation request API.

Replace with a summary that reflects the actual surface: two async generation
methods, capability introspection, file and in-memory media input.

### 0.2 Fix ARCHITECTURE.md

`docs/ARCHITECTURE.md:177-182` repeats the same outdated scaffold language.
Rewrite to describe the implemented multimodal request flow: Godot dictionary
conversion, worker submission, `LlamaMultimodalHandle::evaluate_prompt`,
shared sampling loop.

### 0.3 Verify CHANGELOG

The `[Unreleased]` block already covers the multimodal additions. No changes
needed unless a phase below adds new public surface.

---

## Phase 1 -- Test Coverage

**Goal**: Cover the multimodal paths that currently have no tests.

### 1.1 Unit tests for media input conversion

File: `tests/unit/test_media_input_conversion.cpp`

Test `to_internal_media_inputs` logic (extracted or tested through the Godot
binding seam):

- Dictionary with `path` only.
- Dictionary with `data` only.
- Dictionary with both `path` and `data` (data takes precedence).
- `type` key: `"image"`, `"audio"`, `"voice"`, `"speech"`, missing, garbage.
- `id` key present and absent.
- Empty array, array with non-dictionary entries, array with dictionaries
  missing both `path` and `data`.

### 1.2 Unit tests for multimodal evaluate_prompt validation

Extend `tests/unit/test_multimodal_handle.cpp`:

- Empty prompt with valid media input span.
- Non-empty prompt with empty media input span.
- Null `llama_context` with otherwise valid arguments (when mtmd is compiled in).
- Verify each error code matches the expected `ErrorCode`.

### 1.3 Unit tests for worker multimodal submission

File: `tests/unit/test_worker_multimodal.cpp`

Test `InferenceWorker` multimodal bookkeeping and cancellation paths without a
real model, but do not assume a stopped worker rejects submission unless the
implementation is intentionally tightened first:

- `submit_multimodal_with_id` correctly advances the request ID counter.
- Multimodal request submission preserves prompt, media inputs, and options.
- Cancelling a multimodal request before the worker picks it up results in
  `on_cancelled` once the worker observes the cancelled request.

### 1.4 Integration tests (headless)

File: `tests/integration/test_multimodal_session.gd` (or C++ headless test)

These require a small multimodal model. If no redistributable test model is
available, keep worker-level mocks for fast coverage, but do not treat them as
a substitute for the Godot-facing smoke test. The repo should still keep a
real Godot-level integration target, even if it is opt-in/manual when a model
artifact is unavailable in CI.

- Open session with `LlamaMultimodalConfig` set, verify `opened()` signal.
- `supports_image_input()` returns `true` after open with a vision projector.
- `generate_multimodal_async` with a single image emits `token_emitted` and
  `completed`.
- `generate_multimodal_messages_async` with chat template emits `completed`.
- Mismatched marker count fires `failed` with `InvalidParameter`.
- Missing `multimodal_config` on session fires `failed` with
  `CapabilityUnavailable`.
- `cancel` during multimodal generation fires `cancelled`.
- Open without mtmd build (`GODOT_LLAMA_ENABLE_MTMD=OFF`) and call
  `generate_multimodal_async` fires `failed` with `CapabilityUnavailable`.

---

## Phase 2 -- Multimodal Demo Scene

**Goal**: Ship a minimal working example that users can run.

### 2.1 Demo scene

Add `demo/multimodal_main.gd` and `demo/multimodal_main.tscn`:

- UI with: model path input, mmproj path input, image file path input, prompt
  text input, generate button, cancel button, streaming output label, status
  label.
- On "Open": create `LlamaModelConfig` with `LlamaMultimodalConfig`, call
  `open()`, poll each frame, and wait for `opened()` before enabling
  generation.
- On "Generate": send `generate_multimodal_async` with the image and a prompt
  containing `<__media__>`.
- Stream tokens into the output label.
- Show `supports_image_input()` / `supports_audio_input()` status after open.
- Handle failure, cancellation, and close.

The demo should explicitly model the shipped session contract rather than
teaching a synchronous open-then-generate flow.

### 2.2 Demo documentation

Add a "Multimodal Demo" section to `README.md` or `docs/BUILD.md` explaining:

- Which models work (any GGUF + matching mmproj, e.g. LLaVA, Qwen-VL, Gemma).
- How to obtain model + projector files.
- How to run the demo.

---

## Phase 3 -- Quality-of-Life Improvements

**Goal**: Make the API more ergonomic and robust without expanding scope.

### 3.1 Early marker-count validation

In `LlamaSession::generate_multimodal_async` and
`generate_multimodal_messages_async`, count occurrences of the configured
`media_marker` in the prompt string and compare with `media_inputs.size()`.
If mismatched, return a clear error immediately from the Godot layer instead
of waiting for `mtmd_tokenize` to return error code 1 on the worker thread.

This gives synchronous feedback at submission time when the mismatch is
detectable (raw prompt path). For the messages path, the prompt is only known
after template expansion, so the check runs post-template but pre-submit.

### 3.2 Godot Image convenience helper

Add a static or utility method on `LlamaSession` (or a free function in a
helper class):

```gdscript
## Returns a media input dictionary from a Godot Image.
static func image_to_media_input(image: Image) -> Dictionary
```

Implementation: convert `Image` to PNG bytes via `image.save_png_to_buffer()`,
return `{"data": bytes, "type": "image"}`.

This avoids users having to figure out the serialization themselves. Keep it
as a GDScript utility or a bound static method. Do not let this convenience
shape the broader multimodal API around image-only assumptions; audio remains a
first-class input type, and future media types should still fit the same
dictionary/resource model cleanly.

### 3.3 Improve error messages

- When `evaluate_prompt` gets tokenize error 1 (marker mismatch), include the
  marker string and both counts (expected vs. found) in the error message.
- When a media file fails to load, include the file extension and suggest
  supported formats.

### 3.4 Validate media file readability at submission

In `to_internal_media_inputs`, when `path` is set and `data` is empty, check
that the file exists and is readable before enqueueing. Return a validation
error from the Godot layer instead of deferring the failure to the worker.

---

## Phase 4 -- Advanced Features

**Goal**: Expose upstream capabilities that materially improve production use.

### 4.1 Cancellation during media preprocessing

**Problem**: `LlamaMultimodalHandle::evaluate_prompt` runs synchronously in the
worker thread. A large image or audio file can block the worker for seconds
during `mtmd_tokenize` and `mtmd_helper_eval_chunks` with no cancellation
check.

**Solution**: Break `evaluate_prompt` into stages:

1. Load bitmaps (check cancellation after each).
2. Tokenize (single call, no mid-call cancellation possible in libmtmd).
3. Evaluate chunks (use `mtmd_helper_eval_chunk_single` per chunk instead of
   the bulk helper, checking cancellation between chunks).

Pass a `const std::atomic<bool> &cancelled` to `evaluate_prompt`. In
`process_request`, pass `req.cancelled`.

### 4.2 Progress signal for media encoding

**Problem**: Users have no feedback while a large image is being encoded.

**Solution**: Wire `cb_eval` from `mtmd_context_params`:

1. Add a request-scoped callback path such as
   `RequestCallbacks::on_media_progress`.
2. In `LlamaMultimodalHandle::create`, set `params.cb_eval` to a function
   that forwards progress into the active request path.
3. Expose via a new `media_encoding_progress(request_id: int, progress: float)`
   signal on `LlamaSession`.
4. Emit on the main thread through the existing event queue.

Do not model progress as session-global config state. The Godot layer needs to
attribute progress to the active request explicitly.

### 4.3 Warmup encoding pass

Expose `warmup` from `mtmd_context_params` as
`LlamaMultimodalConfig.warmup_on_init` (default `false`).

When enabled, `mtmd_init_from_file` runs a dummy encoding pass during
initialization. This moves first-request latency into `open()` time where it
is expected.

### 4.4 Multi-turn media caching

**Problem**: Every multimodal request clears the KV cache and re-encodes all
media. This is wasteful for multi-turn conversations where the same image
appears in every turn.

**Design sketch** (investigate before committing):

1. Track bitmap IDs in a cache map (`id -> kv_cache_slot`).
2. On `generate_multimodal_async`, compare incoming IDs against cache.
3. For cache hits, skip re-encoding and adjust `n_past`.
4. For cache misses, encode and register.
5. Expose a `clear_media_cache()` method on `LlamaSession`.

This requires understanding `libmtmd`'s chunk-level copy API
(`mtmd_input_chunk_copy`) and whether `llama.cpp`'s KV cache supports
partial preservation across requests. Mark as experimental until validated.

If pursued, keep the cache representation media-generic rather than
image-specific so the same infrastructure can support audio inputs and future
media families without redesigning the public surface.

### 4.5 Flash attention for multimodal projector

Expose `flash_attn_type` from `mtmd_context_params` as
`LlamaMultimodalConfig.flash_attn_type`:

- `-1` = auto (default)
- `0` = disabled
- `1` = enabled

Mirror the same semantics as `LlamaModelConfig.flash_attn_type`.

---

## Phase 5 -- Upstream Awareness

**Goal**: Surface upstream model metadata that affects correctness.

### 5.1 M-RoPE and non-causal attention introspection

Some multimodal models (Qwen-VL, GLM4V) use M-RoPE position encoding or
require non-causal attention masks during media decoding.
`mtmd_helper_eval_chunks` handles this internally, but users may want to query
these capabilities.

Add read-only introspection methods:

```cpp
bool uses_mrope() const noexcept;           // wraps mtmd_decode_use_mrope()
bool uses_non_causal_decode() const noexcept; // wraps mtmd_decode_use_non_causal()
```

Bind as `LlamaSession.uses_multimodal_mrope()` and
`LlamaSession.uses_multimodal_non_causal_decode()`. Document as informational
(godorama handles the details internally).

### 5.2 Chunk token count query

After multimodal tokenization, the token cost of an image/audio input is
model-dependent (can range from ~50 to ~4000 tokens). Expose:

```gdscript
## Returns the multimodal token count recorded for a completed request.
func get_multimodal_token_count(request_id: int) -> int
```

This helps users budget their context window when mixing text and media.

Implementation options, in preferred order:

1. Include `multimodal_token_count` directly in the existing `completed` stats
   dictionary for multimodal requests.
2. If a query API is still needed, store counts keyed by `request_id` and
   expire them with the rest of the request lifecycle.

Do not expose session-global "last request" accounting in an async API that can
queue multiple requests.

---

## Phase Order and Dependencies

```
Phase 0  (docs sync)         -- no code dependencies, do first
Phase 1  (tests)             -- depends on Phase 0 for accurate docs
Phase 2  (demo)              -- depends on Phase 0 for accurate contract docs
Phase 3  (quality-of-life)   -- independent of Phases 1-2
Phase 4  (advanced features) -- depends on Phase 1 for test coverage
Phase 5  (upstream awareness) -- depends on Phase 3 for clean error model

Recommended execution order:
  0 -> 1 -> 2 (parallel with 3) -> 4 -> 5
```

---

## Out of Scope

These are intentionally excluded from this plan:

- **Video input**: `libmtmd` does not support video. Upstream has a TODO for
  it but no implementation.
- **Streaming audio input**: `libmtmd` has `mtmd_audio_streaming_istft` but
  this is an audio reconstruction primitive, not a streaming input pipeline.
- **Multimodal embeddings**: Not supported by `libmtmd` at the multimodal
  level. Text-only `embed()` remains the embeddings path.
- **Multimodal RAG**: Integrating vision/audio into the RAG pipeline
  (e.g., image-based retrieval) is a separate initiative.
- **Speech-to-text via multimodal**: While audio input is supported, using it
  as a speech recognition pipeline requires model-specific prompt engineering
  that is outside the scope of the extension API.
- **Custom image preprocessors**: `libmtmd` selects the preprocessor based on
  the mmproj metadata. Exposing preprocessor selection is unnecessary churn.

These remain future-facing design constraints even though they are out of
scope for this plan:

- Do not block future multimodal RAG work by coupling retrieval-oriented media
  metadata to image-only helpers or session-global caches.
- Do not block richer audio support by assuming multimodal inputs are still
  images-with-extra-flags.
- Do not block eventual video support by baking "single still image per marker"
  semantics too deeply into Godot-facing API names or internal cache/accounting
  structures.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `libmtmd` API is marked experimental; upstream may break it | Medium | High | All mtmd calls are behind conditional compilation and a single wrapper class. Pin the llama.cpp revision. |
| No small redistributable multimodal model for CI | High | Medium | Keep worker-level mocks for coverage, but retain a real Godot-level smoke target and a manual integration path that accepts model/projector arguments. |
| Multi-turn caching (Phase 4.4) may not work with llama.cpp KV cache semantics | Medium | Low | Mark as experimental. Validate with at least two model families before stabilizing. |
| `cb_eval` progress callback granularity may be too coarse for useful progress reporting | Low | Low | Test with a real model. Fall back to a binary "encoding started" / "encoding finished" signal if needed. |

---

## Definition of Done (per phase)

- [ ] Clean build with `cmake --preset dev && cmake --build --preset build-dev`
- [ ] All unit tests pass: `ctest --preset test-dev --output-on-failure`
- [ ] No new main-thread blocking
- [ ] Public API remains Variant-compatible
- [ ] Error messages are actionable
- [ ] `docs/API.md` and `docs/ARCHITECTURE.md` updated if API changed
- [ ] CHANGELOG updated if public surface changed
- [ ] `.gdextension` unaffected (no binary renames)
