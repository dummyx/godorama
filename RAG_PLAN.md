# libSQL Local-Only Migration Plan

## Context

The previous `sqlite-vector` plan is retired.

It had three problems for this repository:

1. licensing risk for non-open-source packaging
2. incorrect assumptions about filtered top-k behavior
3. quantized-index lifecycle complexity for a mutable corpus

`libSQL` is a better fit because it stays in-process, keeps SQLite file-format
and API compatibility, and provides built-in vector types, distance functions,
and vector indexes.

This plan adopts `libSQL` as a **local embedded library only**.

It does **not** adopt:

- Turso Cloud
- sync URLs
- auth tokens
- embedded replicas
- background network activity

The migration must preserve the repository constraints:

- local-first and offline by default
- deterministic builds with pinned revisions
- no hidden network access
- exact retrieval remains the default behavior
- public Godot API stays small and Variant-compatible

## Compatibility assumption

RAG APIs are currently unused.

This migration does **not** need to preserve RAG API compatibility across:

- Godot method names
- retrieval option keys
- schema details
- persisted corpus format
- support for legacy retrieval modes that complicate the backend

The libSQL migration should prefer the smallest coherent API surface for the
new backend, even when that is a breaking change, as long as docs and tests are
updated in the same change.

## Design goals

1. Replace the bundled SQLite engine with a pinned local `libSQL` build.
2. Keep the storage layer using the SQLite-compatible `sqlite3_*` C API.
3. Move exact cosine retrieval into SQL so we stop loading the whole corpus
   into memory for scoring.
4. Add libSQL vector indexes as an optional acceleration path, not as the only
   retrieval path.
5. Preserve current filtering semantics for `source_ids`, exclusions, and
   metadata filters.
6. Avoid new main-thread blocking during migration or index maintenance.

## Non-goals

- No Turso Cloud integration.
- No remote replication or sync.
- No dependency on preview `libsql_*` client APIs in the hot path.
- No silent change from exact retrieval to approximate retrieval.
- No carrying forward unused RAG surface area just for compatibility.

## Retrieval contract to preserve

Current RAG docs describe retrieval as exact dense search with filtering,
deduplication, and optional MMR.

The libSQL migration must preserve that default:

- exact search remains the default
- filtering happens before the final top-k result is chosen
- deduplication and MMR remain in C++
- ANN is opt-in and may overfetch, but must fall back to exact search when it
  cannot satisfy the current filtered contract

## Phase 0: libSQL Spike

**Goal**: prove the local-only integration shape before committing the repo to
the dependency swap.

**Deliverables**

1. Build a pinned `libSQL` revision as a local static dependency.
2. Verify the repository can continue using `sqlite3_open`,
   `sqlite3_prepare_v2`, `sqlite3_step`, and friends unchanged.
3. Verify local-file operation with no remote URL, token, or sync settings.
4. Verify libSQL vector SQL on a local DB:
   - `vector32(...)`
   - `vector_distance_cos(...)`
   - `vector_distance_l2(...)`
   - `libsql_vector_idx(...)`
   - `vector_top_k(...)`
5. Verify index maintenance behavior on insert, update, and delete.
6. Determine the exact vector storage shape to use in the schema:
   - whether `F32_BLOB` is sufficient
   - or whether the chosen revision requires `F32_BLOB(<dim>)`
7. Verify this on the minimum CI matrix:
   - Ubuntu
   - macOS
   - Windows

**Decision rule**

- If the local embedded engine path is stable and SQLite-compatible on all
  target platforms, continue.
- If the implementation would require repo-wide dependence on preview client
  bindings or remote-only features, stop and keep SQLite.

## Phase 1: Dependency Integration

**Goal**: swap the underlying embedded engine from bundled SQLite to pinned
local `libSQL`.

### Dependency policy

- Add `thirdparty/libsql` as a pinned submodule or pinned vendored source tree.
- Record the exact revision in docs and CHANGELOG.
- Do not mix this dependency update with unrelated work.

### Build-system strategy

The current repo aliases the bundled SQLite build as `SQLite::SQLite3`.
Keep that alias, but make it point to the local libSQL-backed target instead.

This keeps most store code and test code unchanged while the engine changes
underneath.

### CMake plan

1. Add a new static target for the chosen libSQL core.
2. Expose the same include surface currently used by `src/core/rag`.
3. Preserve:
   - `SQLite::SQLite3`
   - out-of-source builds
   - `compile_commands.json`
   - warnings-as-errors behavior
4. Do not enable remote protocol or sync-specific features in the runtime
   dependency path.

### Why not adopt the `libsql_*` C client API

The local embedded engine path is lower risk for this repo because:

- the current code already uses `sqlite3_*`
- the repo already treats SQLite as an embedded library
- this avoids coupling core runtime behavior to preview client-SDK surfaces
- it keeps the migration mostly in `CMakeLists.txt` and the RAG SQL layer

## Phase 2: Schema Migration

**Goal**: move the chunk embedding column to a libSQL vector-aware schema.

### Key point

Do **not** assume the current raw float BLOB layout can be reinterpreted
in-place as a libSQL vector column.

The safe migration is to rebuild stored embeddings into the new format.

### Schema version

Bump `kSchemaVersion` from `1` to `2`.

### New schema shape

Replace the raw embedding column with a libSQL vector-aware column:

```sql
CREATE TABLE chunks(
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    source_version TEXT NOT NULL,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    display_text TEXT NOT NULL,
    metadata_blob TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    byte_start INTEGER NOT NULL,
    byte_end INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    embedding_fingerprint TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    embedding_normalized INTEGER NOT NULL,
    vector_metric TEXT NOT NULL,
    pooling_type INTEGER NOT NULL,
    embedding_vec F32_BLOB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);
```

If the pinned libSQL revision requires a dimension-qualified declaration for
indexed vectors, resolve that in Phase 0 and use the verified type spelling in
the schema implementation.

### Migration strategy

Do not run a full re-embedding migration inside `open()`.

Instead:

1. preserve `sources` and `source_metadata`
2. create the schema-v2 tables
3. mark embeddings as stale in `rag_meta`
4. rebuild chunk embeddings asynchronously from stored source text

Because `sources.normalized_text` is already persisted, the repo can rebuild
deterministically without downloading anything or requiring original source
files to still exist.

### `rag_meta` additions

Add:

- `embedding_storage_format`
- `ann_index_ready`
- `ann_index_metric`

Suggested values:

- `embedding_storage_format=libsql_f32`
- `ann_index_ready=0|1`
- `ann_index_metric=cosine`

## Phase 3: Write Path Changes

**Goal**: write embeddings in libSQL vector format during ingestion.

### File

`src/core/rag/sqlite_corpus_store.cpp`

### Insert path

Replace raw embedding BLOB inserts with libSQL vector conversion at write time.

The implementation chosen in Phase 0 must be one of:

1. bind a validated libSQL-native vector blob directly
2. call `vector32(?)` on a bound value that the pinned libSQL revision accepts
3. as a last resort, call `vector32(?)` on a generated JSON array string during
   ingestion only

The migration should prefer option 1 or 2.

Do not add JSON serialization inside retrieval hot loops.

### Validation

At insert time, validate:

- embedding dimensions match `embedding_dimensions`
- metric and normalization metadata match corpus state
- zero-length embeddings are rejected as storage corruption

## Phase 4: Exact Retrieval in SQL

**Goal**: keep exact retrieval as the default while moving cosine scoring into
SQL.

### Why this is the default path

Exact SQL retrieval fixes the biggest problem in the current implementation:
we no longer need to load every candidate embedding into memory just to compute
query similarity.

It also preserves filtered top-k semantics because filters stay in the main SQL
query instead of being applied after a separate ANN primitive has already chosen
neighbors.

### New store method

Add a cosine exact-search method to `CorpusStore`:

```cpp
struct VectorSearchHit {
    std::string chunk_id;
    float distance = 0.0f;
};

[[nodiscard]] virtual Error exact_vector_search(
    const std::vector<float> &query_vector,
    const RetrievalOptions &options,
    std::vector<VectorSearchHit> &out_hits) const = 0;
```

Keep `fetch_chunks_by_ids()` for metadata and optional embedding fetches used by
MMR.

### Exact cosine SQL shape

```sql
SELECT
    c.chunk_id,
    vector_distance_cos(c.embedding_vec, ?1) AS distance
FROM chunks AS c
WHERE 1=1
  [AND source filters]
  [AND metadata filters]
ORDER BY distance ASC, c.source_id ASC, c.chunk_index ASC
LIMIT ?2;
```

Where:

- `?1` is the query embedding encoded as a libSQL vector
- `?2` is `candidate_k`

### Determinism

Always include a stable tie-break after distance so tests stay deterministic.

### Retriever rewrite

For the default retrieval mode:

1. embed query
2. call `store.exact_vector_search(...)`
3. fetch metadata for the selected chunk ids
4. convert cosine distance to similarity with `1.0f - distance`
5. keep deduplication, reranking, and MMR behavior unchanged

### Metric scope

libSQL’s documented vector surface exposes cosine and L2 distance.

Because RAG API compatibility is not required, the first libSQL migration
should intentionally narrow the RAG surface to **cosine retrieval only**.

Rules:

- remove `dot` from the RAG-facing config and docs
- keep one exact cosine retrieval path
- keep one optional cosine ANN path
- do not ship a separate compatibility fallback for `dot`

This simplifies the backend, tests, and docs considerably.

## Phase 5: Optional ANN Index Path

**Goal**: add libSQL vector indexes as an opt-in acceleration path.

### Why optional

libSQL vector indexes are approximate nearest-neighbor indexes.
They are useful, but they are not equivalent to the current exact retrieval
contract.

### Index creation

Create an index only for cosine corpora:

```sql
CREATE INDEX idx_chunks_embedding_ann
ON chunks(libsql_vector_idx(embedding_vec));
```

If the chosen revision needs explicit settings:

```sql
CREATE INDEX idx_chunks_embedding_ann
ON chunks(libsql_vector_idx(embedding_vec, 'metric=cosine'));
```

### Runtime option

Add to `RetrievalOptions`:

```cpp
bool use_ann_search = false;
```

Default stays `false`.

### ANN query shape

```sql
SELECT c.chunk_id, v.distance
FROM vector_top_k('idx_chunks_embedding_ann', ?1, ?2) AS v
JOIN chunks AS c ON c.rowid = v.id
WHERE 1=1
  [AND source filters]
  [AND metadata filters]
ORDER BY v.distance ASC;
```

### Filter correctness rule

Because `vector_top_k(...)` chooses neighbors before outer filters are applied,
ANN retrieval must not assume the first query is sufficient.

Implementation rule:

1. overfetch from ANN using a bounded multiplier
2. apply normal filters
3. if filtered hits are still insufficient, fall back to exact SQL retrieval

This preserves current behavior without silently returning too few or wrong
results.

### Index lifecycle

Unlike the retired `sqlite-vector` plan, libSQL vector indexes are part of the
engine and are updated automatically with base-table writes.

Still verify this explicitly in tests for:

- insert
- delete
- rebuild
- clear

### Godot-facing option

Expose `use_ann_search` in the retrieval options dictionary.

Document clearly:

- `false` = exact search
- `true` = approximate cosine search with exact fallback when needed to satisfy
  filtering semantics

## Phase 6: CorpusStore and Retriever Refactor

**Goal**: keep layering clean while supporting the two retrieval paths.

### `CorpusStore`

Add:

- `exact_vector_search(...)`
- `fetch_chunks_by_ids(...)`
- `ensure_ann_index(...)`
- `drop_ann_index(...)`
- `is_ann_index_ready(...)`

Do not expose libSQL-specific types outside the store implementation.

### `DenseRetriever`

Refactor retrieval into two modes:

1. `cosine + exact`
2. `cosine + ann`

Keep:

- overlap suppression
- reranker hook
- MMR
- existing `RetrievalStats` fields

### Stats additions

Add:

- `search_mode` with values like `exact_sql` and `ann_sql`
- `ann_fallback_used`

Keep existing counts meaningful:

- `scanned_chunks` should represent the number of rows considered by the chosen
  mode when it is knowable
- `candidate_chunks` remains post-threshold, pre-dedup

## Phase 7: Godot API and Runtime Behavior

**Goal**: expose the smallest Godot-facing API that matches the new backend.

### `RagCorpus`

Do not add network or sync-related API.

Godot-facing changes are allowed to be breaking if they simplify the surface.

Add only:

- retrieval option `use_ann_search`
- optional maintenance method if explicit index creation is required:
  `rebuild_vector_index_async()`

Keep index build and rebuild work off the main thread.

### Config cleanup

If `vector_metric` remains exposed at all, reduce it to a single supported
value:

- `cosine`

Prefer removing the option entirely if that produces a cleaner API.

### Open behavior

`open()` must not attempt:

- remote connection
- sync
- full corpus re-embedding migration
- eager ANN rebuild

If a corpus needs migration, surface that as:

- `stale_embeddings = true`
- actionable error details
- async rebuild path

## Phase 8: Tests and Evaluation

**Goal**: verify local-only correctness, exact-search parity, and cross-platform
build behavior.

### New unit tests

1. `test_libsql_local_store.cpp`
   - open local DB
   - schema init
   - source and chunk writes
   - exact cosine query ordering

2. `test_rag_exact_sql_retriever.cpp`
   - exact cosine retrieval with source filters
   - exact cosine retrieval with metadata filters
   - deterministic tie-break ordering

3. `test_rag_ann_retriever.cpp`
   - ANN path returns results
   - ANN overfetch + fallback preserves filtered semantics
   - ANN path remains opt-in and deterministic under test fixtures

4. `test_rag_migration.cpp`
   - schema-v1 DB opens under libSQL
   - stale state reported
   - async rebuild produces schema-v2 vector rows

### Integration tests

Update `tests/integration/test_rag_pipeline.cpp` to cover:

- local libSQL open
- upsert -> retrieve exact path
- optional ANN retrieval smoke test
- migration failure path

### Evaluation updates

Update `docs/RAG_EVALUATION.md` to report both:

- exact SQL retrieval latency
- ANN retrieval latency and recall against exact baseline

ANN evaluation must report recall against exact retrieval instead of claiming a
fixed quality number from upstream docs.

## Phase 9: Documentation and CHANGELOG

### `docs/ARCHITECTURE.md`

Update:

- dependency list: bundled SQLite -> libSQL
- retrieval section: exact SQL cosine path
- optional ANN index path and fallback rules

### `docs/API.md`

Document:

- `use_ann_search`
- exact vs approximate behavior
- cosine-only retrieval
- any removed RAG options as breaking changes

### `README.md`

Update feature text to say:

- local embedded libSQL backend
- exact cosine retrieval in SQL
- optional ANN acceleration for cosine corpora
- no cloud dependency

### `CHANGELOG.md`

Under `[Unreleased]`:

- Added local libSQL backend for RAG storage
- Changed exact cosine retrieval to run in SQL
- Added optional ANN retrieval path for cosine corpora
- Added schema-v2 vector storage and async migration flow

## Implementation order

Ship this in reviewable increments:

1. Phase 0 spike and dependency pin
2. build integration with `SQLite::SQLite3` backed by libSQL
3. schema-v2 vector storage and stale-migration path
4. exact cosine SQL retrieval
5. optional ANN index path
6. docs, evaluation, and cleanup

## Exit criteria

Do not call the migration complete until all of the following are true:

- local libSQL build works on Ubuntu, macOS, and Windows
- no runtime path requires network access
- exact retrieval remains the default behavior
- filtered retrieval semantics are preserved
- the final RAG API surface is documented and intentionally smaller
- ANN is clearly opt-in
- schema migration avoids new main-thread blocking
- docs and CHANGELOG match the shipped API and runtime behavior
