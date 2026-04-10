# RAG Architecture

## Overview

The RAG subsystem is additive to the existing inference stack:

- `RagCorpusConfig` defines storage, chunking, and embedding settings.
- `RagCorpus` is the Godot-facing async wrapper around a shared `rag::CorpusEngine`.
- `RagAnswerSession` composes corpus retrieval with the existing generation worker.

The core RAG path is:

1. normalize UTF-8 text or file input
2. chunk deterministically with token-count enforcement
3. embed chunks with a dedicated embedding context
4. persist source/chunk rows and embeddings in embedded libSQL
5. retrieve with exact cosine SQL search, filter, dedupe, and MMR
6. pack grounded context under a generation-token budget
7. stream answer tokens through `InferenceWorker`

## Core interfaces

- `Chunker`
- `Embedder`
- `CorpusStore`
- `Retriever`
- `Reranker`
- `ContextPacker`

The shipped concrete implementations are:

- `DeterministicChunker`
- `LlamaEmbedder`
- `LibSqlCorpusStore`
- `DenseRetriever`
- `NoopReranker`
- `GroundedContextPacker`

## Persistence

The corpus store uses a local embedded libSQL database with schema version `2`.

Stored entities:

- `sources`
- `source_metadata`
- `chunks`
- `rag_meta`

`rag_meta` persists:

- `schema_version`
- `embedding_fingerprint`
- `embedding_dimensions`
- `embedding_normalized`
- `embedding_storage_format`
- `vector_metric`
- `pooling_type`
- `ann_index_ready`
- `ann_index_metric`

Chunk rows persist:

- stable `chunk_id`
- `source_id`
- `source_version`
- normalized text and display text
- byte and char offsets
- token count
- embedding fingerprint and dimensions
- vector metric and normalization flag
- libSQL `F32_BLOB` vector storage in `embedding_vec`

## Retrieval

The current retriever is exact cosine SQL search:

- query embedding through `LlamaEmbedder` or `MockEmbedder`
- `vector_distance_cos(embedding_vec, ?)` ordering inside libSQL
- normalized higher-is-better relevance score
- overlap suppression for near-duplicate chunks
- optional MMR diversification
- optional reranker hook with explicit status reporting

## Prompt assembly

`GroundedContextPacker`:

- budgets context with the generation tokenizer
- skips overlapping packed chunks
- keeps citations aligned to the chunks actually packed
- uses the model chat template when available, otherwise falls back to a plain prompt
- abstains when no grounded context survives packing
