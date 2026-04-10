# API Reference

## LlamaLoraAdapterConfig (Resource)

Editor-facing LoRA adapter configuration.

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `adapter_path` | `String` | `""` | Path to a LoRA GGUF adapter |
| `scale` | `float` | `1.0` | Adapter scale passed to `llama_set_adapters_lora()` |

## LlamaMultimodalConfig (Resource)

Editor-facing multimodal projector configuration. Assigned to `LlamaModelConfig.multimodal_config` to enable image and audio input through `libmtmd`.

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `mmproj_path` | `String` | `""` | Path to the multimodal projector GGUF |
| `media_marker` | `String` | `"<__media__>"` | Marker string used in prompts to indicate where media should be inserted. Must match what the model expects. |
| `use_gpu` | `bool` | `false` | Whether `libmtmd` may offload projector work to GPU when a backend is enabled |
| `print_timings` | `bool` | `false` | Enable `libmtmd` timing logs |
| `n_threads` | `int` | `-1` | `libmtmd` worker threads (`-1` = default behavior) |
| `image_min_tokens` | `int` | `0` | Lower bound for dynamic-resolution image tokenization (`0` = metadata/default) |
| `image_max_tokens` | `int` | `0` | Upper bound for dynamic-resolution image tokenization (`0` = metadata/default) |

## LlamaModelConfig (Resource)

Configuration resource for model loading. Can be created in the editor or from code.

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model_path` | `String` | `""` | Path to `.gguf` model file |
| `n_ctx` | `int` | `2048` | Context window size in tokens |
| `n_threads` | `int` | `-1` | CPU threads (-1 = auto-detect) |
| `n_batch` | `int` | `512` | Batch size for prompt processing |
| `n_gpu_layers` | `int` | `0` | Layers to offload to GPU (0 = CPU only) |
| `seed` | `int` | `-1` | Random seed (0xFFFFFFFF = random) |
| `use_mmap` | `bool` | `true` | Memory-map model file |
| `use_mlock` | `bool` | `false` | Lock model in RAM |
| `embeddings_enabled` | `bool` | `false` | Enable embedding extraction |
| `disable_thinking` | `bool` | `false` | When using message templating, request the model's non-thinking path if the template supports it |
| `chat_template_override` | `String` | `""` | Override the model chat template used by `generate_messages_async()` |
| `lora_adapters` | `Array[LlamaLoraAdapterConfig]` | `[]` | Ordered LoRA adapter list applied when the session context is created |
| `multimodal_config` | `LlamaMultimodalConfig` | `null` | Optional `libmtmd` projector configuration loaded during `open()` |

## LlamaSession (RefCounted)

Main interface for model interaction.

### Methods

#### `open(config: LlamaModelConfig) -> int`
Loads a model and starts the inference worker.
- **Blocks**: No
- **Thread-safe**: No (call from main thread)
- **Returns**: 0 if model loading was accepted, error code on immediate validation failure

#### `is_opening() -> bool`
Check if a background model load is currently in progress.
- **Blocks**: No
- **Thread-safe**: Yes

#### `close()`
Stops the worker and releases the model.
- **Blocks**: Yes (waits for worker shutdown)
- **Thread-safe**: No

#### `is_open() -> bool`
Check if a model is loaded and the worker is running.
- **Blocks**: No
- **Thread-safe**: Yes

#### `generate_async(prompt: String, options: Dictionary = {}) -> int`
Submits a generation request. Returns a `request_id` for tracking.
- **Blocks**: No
- **Thread-safe**: No (call from main thread)

#### `generate_messages_async(messages: Array, options: Dictionary = {}, add_assistant_turn: bool = true) -> int`
Submits a message-templated generation request. Each message must be a dictionary containing string `role` and `content` keys.
- **Blocks**: No
- **Thread-safe**: No (call from main thread)
- **Notes**:
  - Uses the full `llama.cpp` common-chat Jinja path.
  - Honors `chat_template_override` and `disable_thinking` from `LlamaModelConfig`.

#### `generate_multimodal_async(prompt: String, media_inputs: Array, options: Dictionary = {}) -> int`
Submits a multimodal generation request with image and/or audio inputs. Requires `multimodal_config` to be set in the session's `LlamaModelConfig`.
- **Blocks**: No
- **Thread-safe**: No (call from main thread)
- **Returns**: `request_id` for tracking, or negative error code on validation failure
- **Notes**:
  - The prompt must contain one `<__media__>` marker (or the configured `media_marker`) for each media input.
  - `media_inputs` is an `Array` of dictionaries, each with the keys documented under [Media Input Dictionary](#media-input-dictionary).
  - The Godot layer validates each media dictionary, readable file paths, and marker count before the request is queued.
  - Emits `token_emitted`, `completed`, `failed`, or `cancelled` through `poll()` like other generation methods.

#### `generate_multimodal_messages_async(messages: Array, media_inputs: Array, options: Dictionary = {}, add_assistant_turn: bool = true) -> int`
Submits a message-templated multimodal generation request. Combines `generate_messages_async` chat template processing with multimodal media input.
- **Blocks**: No
- **Thread-safe**: No (call from main thread)
- **Returns**: `request_id` for tracking, or negative error code on validation failure
- **Notes**:
  - The resulting prompt from chat template expansion must contain one media marker per media input.
  - `messages` follows the same `{role, content}` format as `generate_messages_async`.
  - `media_inputs` follows the same format as `generate_multimodal_async`.
  - The expanded prompt is validated against the configured media marker before the request is queued.

#### `image_to_media_input(image: Image) -> Dictionary`
Static convenience helper that converts a Godot `Image` into a multimodal media dictionary using PNG bytes.
- **Blocks**: Yes (fast)
- **Thread-safe**: Yes
- **Returns**: `{"data": PackedByteArray, "type": "image"}` on success, or an empty dictionary on failure

#### `cancel(request_id: int)`
Cancels a pending or in-progress generation request.
- **Blocks**: No
- **Thread-safe**: Yes

#### `tokenize(text: String, add_bos: bool = false, special: bool = false) -> PackedInt32Array`
Converts text to token IDs.
- **Blocks**: Yes (fast)
- **Thread-safe**: Yes

#### `detokenize(tokens: PackedInt32Array) -> String`
Converts token IDs back to text.
- **Blocks**: Yes (fast)
- **Thread-safe**: Yes

#### `embed(text: String) -> PackedFloat32Array`
Computes embeddings for text. Requires `embeddings_enabled = true` in config.
- **Blocks**: Yes
- **Thread-safe**: No

#### `get_lora_adapter_count() -> int`
Returns the number of LoRA adapters loaded into the currently open session.
- **Blocks**: No
- **Thread-safe**: Yes

#### `supports_image_input() -> bool`
Returns whether the currently loaded multimodal projector reports image support.
- **Blocks**: No
- **Thread-safe**: Yes

#### `supports_audio_input() -> bool`
Returns whether the currently loaded multimodal projector reports audio support.
- **Blocks**: No
- **Thread-safe**: Yes

#### `get_audio_input_sample_rate_hz() -> int`
Returns the multimodal audio sample rate in Hz, or `-1` when audio input is unavailable.
- **Blocks**: No
- **Thread-safe**: Yes

#### `get_multimodal_token_count(request_id: int) -> int`
Returns the stored multimodal token count for a completed multimodal request.
- **Blocks**: No
- **Thread-safe**: Yes
- **Returns**: token count for the completed request, or `-1` when no completed multimodal record is available for that `request_id`

#### `poll()`
Flushes queued events from the worker thread and emits signals.
Must be called each frame (e.g., from `_process`).
- **Blocks**: No
- **Thread-safe**: No (main thread only)

### Notes

- `open()` is asynchronous. `opened()` or `failed()` arrives later through `poll()`.
- `disable_thinking` only applies to the message-templated path. It does not rewrite raw prompt text passed to `generate_async()`.
- `lora_adapters` uses the stable `llama.cpp` adapter API and is applied to the session context at open time.
- `multimodal_config` loads `libmtmd` and enables multimodal generation through `generate_multimodal_async` and `generate_multimodal_messages_async`.

### Media Input Dictionary

Each element of the `media_inputs` array passed to `generate_multimodal_async` or `generate_multimodal_messages_async` is a `Dictionary` with the following keys:

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `path` | `String` | Yes (or `data`) | `""` | File path to an image or audio file |
| `data` | `PackedByteArray` | Yes (or `path`) | empty | In-memory image/audio bytes. When non-empty, takes precedence over `path` |
| `type` | `String` | No | `"image"` | Media type: `"image"` or `"audio"` (also accepts `"voice"`, `"speech"`) |
| `id` | `String` | No | `""` | Optional identifier for KV cache tracking |

Supported image formats: JPG, PNG, BMP, GIF, and others supported by `stb_image`.
Supported audio formats: WAV, MP3, FLAC, and others supported by `miniaudio`.

Invalid entries are rejected immediately at submission time. Each array element must be a `Dictionary`, `path` must point to a readable file when `data` is empty, and `data` must be a non-empty `PackedByteArray` if provided.

### Generation Options Dictionary

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_tokens` | `int` | `256` | Maximum tokens to generate |
| `temperature` | `float` | `0.8` | Sampling temperature |
| `top_p` | `float` | `0.95` | Nucleus sampling threshold |
| `top_k` | `int` | `40` | Top-K sampling |
| `min_p` | `float` | `0.05` | Min-P sampling threshold |
| `repeat_penalty` | `float` | `1.1` | Repetition penalty |
| `seed_override` | `int` | none | Override session seed for this request |
| `stop` | `Array[String]` | `[]` | Stop sequences |

### Signals

#### `opened()`
Emitted when the model has been loaded and the worker is ready.

#### `token_emitted(request_id: int, token_text: String, token_id: int)`
Emitted for each generated token during streaming.

#### `completed(request_id: int, text: String, stats: Dictionary)`
Emitted when generation completes. `stats` contains:
- `tokens_generated: int`
- `time_ms: float`
- `tokens_per_second: float`
- `multimodal_token_count: int` for multimodal requests

#### `failed(request_id: int, error_code: int, error_message: String, details: String)`
Emitted on generation failure.

#### `cancelled(request_id: int)`
Emitted when a generation request is cancelled.

## LlamaEvalSession (RefCounted)

Async eval/prefill interface for callers that already have embedding inputs.

### Methods

#### `open(config: LlamaModelConfig) -> int`
Loads a model/context asynchronously.
- **Blocks**: No
- **Thread-safe**: No

#### `close()`
Stops the eval worker and releases the model/context.
- **Blocks**: Yes
- **Thread-safe**: No

#### `is_open() -> bool`
- **Blocks**: No
- **Thread-safe**: Yes

#### `is_opening() -> bool`
- **Blocks**: No
- **Thread-safe**: Yes

#### `run_prefill_async(inputs_embeds: PackedFloat32Array, sequence_length: int, position_ids: PackedInt32Array = [], position_components: int = 1, logit_start: int = 0, logit_count: int = 0, include_hidden_state: bool = true, clear_kv_cache: bool = true) -> int`
Queues an embedding-prefill request and returns a `request_id`.
- **Blocks**: No
- **Thread-safe**: No
- **Notes**:
  - `inputs_embeds` must match `sequence_length * hidden_size`.
  - `position_ids` may contain one or more position components per token. When only one base position per token is provided, `godorama` auto-expands layouts for 1-, 3-, and 4-component models.
  - `logit_start` + `logit_count` select a returned slice of the final-step logits.
  - `clear_kv_cache = true` makes each request independent by default.

#### `cancel(request_id: int)`
Cancels a queued or active eval request.
- **Blocks**: No
- **Thread-safe**: Yes

#### `poll()`
Flushes queued `opened` / `completed` / `failed` / `cancelled` signals to the main thread.
- **Blocks**: No
- **Thread-safe**: No

### Signals

#### `opened()`
Emitted when the model/context is ready.

#### `completed(request_id: int, result: Dictionary)`
`result` currently contains:
- `logits: PackedFloat32Array`
- `logits_shape: PackedInt64Array`
- `hidden_states: PackedFloat32Array` when `include_hidden_state = true`
- `hidden_states_shape: PackedInt64Array` when `include_hidden_state = true`

#### `failed(request_id: int, error_code: int, error_message: String, details: String)`
Emitted on request failure or asynchronous open failure.

#### `cancelled(request_id: int)`
Emitted when a request is cancelled.

## RAG Additions

### `RagCorpusConfig`

Key properties:

- `storage_path`
- `chunk_size_tokens`
- `chunk_overlap_tokens`
- `normalize_embeddings`
- `max_batch_texts`
- `embedding_model_path`
- `embedding_n_ctx`
- `embedding_n_threads`
- `enable_reranker`
- `reranker_model_path`
- `parser_mode`
- `supported_extensions`

### `RagCorpus`

Key methods:

- `open(config: RagCorpusConfig) -> int`
- `close()`
- `is_open() -> bool`
- `upsert_text_async(source_id: String, text: String, metadata := {}) -> int`
- `upsert_file_async(path: String, metadata := {}) -> int`
- `delete_source_async(source_id: String) -> int`
- `clear_async() -> int`
- `rebuild_async() -> int`
- `cancel_job(job_id: int)`
- `retrieve_async(query: String, options := {}) -> int`
- `get_stats() -> Dictionary`
- `poll()`

Signals:

- `ingest_progress(job_id, done, total)`
- `ingest_completed(job_id, stats)`
- `retrieve_completed(request_id, hits, stats)`
- `failed(request_id_or_job_id, error_code, error_message, details)`

Notes:

- RAG retrieval is cosine-only.
- `retrieve_completed(..., stats)` now reports `search_mode`, which is currently `exact_sql` for the embedded libSQL path.

### `RagAnswerSession`

Key methods:

- `open_generation(config: LlamaModelConfig) -> int`
- `close_generation()`
- `is_generation_open() -> bool`
- `answer_async(corpus: RagCorpus, question: String, retrieval_options := {}, generation_options := {}) -> int`
- `cancel(request_id: int)`
- `poll()`

Signals:

- `token_emitted(request_id, token_text, token_id)`
- `completed(request_id, text, citations, stats)`
- `failed(request_id, error_code, error_message, details)`
- `cancelled(request_id)`
