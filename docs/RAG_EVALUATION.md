# RAG Evaluation

## Offline fixture harness

The repository ships an offline executable:

```sh
./build/dev/tests/integration/godot_llama_rag_eval
```

It evaluates a small synthetic corpus with deterministic mock embeddings and records:

- `recall_at_2`
- `mrr`
- `ingest_ms`
- `avg_retrieve_ms`
- `answer_ms`

## Current fixture results

Run date: 2026-04-02

- `recall_at_2 = 1.0`
- `mrr = 1.0`
- `ingest_ms = 14.2097`
- `avg_retrieve_ms = 0.196238`
- `answer_ms = 0`

## Notes

- The evaluation uses the real SQLite-backed `CorpusEngine`.
- The embedder is mocked so the harness is fast, deterministic, and CI-friendly.
- `answer_ms` is `0` in the current fixture harness because it validates retrieval quality and corpus latency, not model-token generation.
