# RAG API

## `RagCorpusConfig`

Editor-facing resource for corpus storage and embedding settings.

Properties:

- `storage_path`: SQLite database path
- `chunk_size_tokens`: hard token budget per chunk
- `chunk_overlap_tokens`: overlap carried into the next chunk
- `normalize_embeddings`: normalize vectors before persistence/retrieval
- `vector_metric`: `cosine` or `dot`
- `max_batch_texts`: maximum texts accepted per embedding call
- `embedding_model_path`: GGUF model path for the embedding role
- `embedding_n_ctx`
- `embedding_n_threads`
- `enable_reranker`
- `reranker_model_path`
- `parser_mode`: `auto`, `text`, or `markdown`
- `supported_extensions`

## `RagCorpus`

`open(config) -> int`

- blocks while opening the embedding model and SQLite corpus
- main-thread only

`upsert_text_async(source_id, text, metadata := {}) -> int`

- non-blocking
- queues deterministic normalization, chunking, embedding, and persistence

`upsert_file_async(path, metadata := {}) -> int`

- non-blocking
- supports `.txt`, `.md`, and `.markdown` by default

`retrieve_async(query, options := {}) -> int`

- non-blocking
- emits structured hits and retrieval stats

Retrieval options:

- `top_k`
- `candidate_k`
- `score_threshold`
- `max_context_chunks`
- `max_context_tokens`
- `source_ids`
- `exclude_source_ids`
- `metadata_filter`
- `use_mmr`
- `use_reranker`

## `RagAnswerSession`

`open_generation(config) -> int`

- blocks while opening the generation model

`answer_async(corpus, question, retrieval_options := {}, generation_options := {}) -> int`

- non-blocking
- performs retrieval and prompt assembly on the answer queue
- streams final generation tokens through the existing inference worker

Signals:

- `token_emitted`
- `completed`
- `failed`
- `cancelled`

`completed` carries:

- generated text
- citations for the packed prompt chunks only
- answer stats including retrieval stats and packing counts
