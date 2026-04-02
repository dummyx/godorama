# API Reference

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
| `chat_template_override` | `String` | `""` | Override model's chat template |

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

#### `poll()`
Flushes queued events from the worker thread and emits signals.
Must be called each frame (e.g., from `_process`).
- **Blocks**: No
- **Thread-safe**: No (main thread only)

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

#### `failed(request_id: int, error_code: int, error_message: String, details: String)`
Emitted on generation failure.

#### `cancelled(request_id: int)`
Emitted when a generation request is cancelled.

## RAG Additions

### `RagCorpusConfig`

Key properties:

- `storage_path`
- `chunk_size_tokens`
- `chunk_overlap_tokens`
- `normalize_embeddings`
- `vector_metric`
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
