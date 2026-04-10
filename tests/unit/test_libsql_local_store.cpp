#include <catch2/catch_test_macros.hpp>

#include "godot_llama/rag/factories.hpp"

#include <sqlite3.h>

#include <array>
#include <filesystem>
#include <string>

using namespace godot_llama::rag;

namespace {

std::filesystem::path unique_db_path(const char *name) {
    const auto base = std::filesystem::temp_directory_path() / "godot_llama_libsql_tests";
    std::filesystem::create_directories(base);
    const auto path = base / name;
    std::filesystem::remove(path);
    return path;
}

SourceRecord make_source(std::string source_id, std::string text, Metadata metadata) {
    SourceRecord source;
    source.source_id = std::move(source_id);
    source.source_version = make_source_version(text);
    source.title = source.source_id;
    source.source_path = source.source_id + ".txt";
    source.normalized_text = std::move(text);
    source.metadata = std::move(metadata);
    source.created_at = "2026-04-10T00:00:00Z";
    source.updated_at = source.created_at;
    return source;
}

ChunkRecord make_chunk(const SourceRecord &source, int32_t chunk_index, int64_t byte_start, int64_t byte_end,
                       std::string display_text, std::vector<float> embedding) {
    ChunkRecord chunk;
    chunk.chunk_id = make_chunk_id(source.source_id, source.source_version, byte_start, byte_end, chunk_index);
    chunk.source_id = source.source_id;
    chunk.source_version = source.source_version;
    chunk.title = source.title;
    chunk.source_path = source.source_path;
    chunk.normalized_text = source.normalized_text;
    chunk.display_text = std::move(display_text);
    chunk.metadata = source.metadata;
    chunk.chunk_index = chunk_index;
    chunk.byte_start = byte_start;
    chunk.byte_end = byte_end;
    chunk.char_start = static_cast<int32_t>(byte_start);
    chunk.char_end = static_cast<int32_t>(byte_end);
    chunk.token_count = 4;
    chunk.embedding_info.model_fingerprint = "fixture";
    chunk.embedding_info.dimensions = static_cast<int32_t>(embedding.size());
    chunk.embedding_info.normalized = true;
    chunk.embedding_info.metric = VectorMetric::Cosine;
    chunk.embedding_info.pooling_type = 0;
    chunk.embedding = std::move(embedding);
    return chunk;
}

void exec_sql(sqlite3 *db, const char *sql) {
    char *error_message = nullptr;
    const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &error_message);
    INFO(std::string(error_message ? error_message : ""));
    REQUIRE(rc == SQLITE_OK);
    sqlite3_free(error_message);
}

void bind_text(sqlite3_stmt *stmt, int index, std::string_view value) {
    sqlite3_bind_text(stmt, index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
}

void bind_blob(sqlite3_stmt *stmt, int index, const std::vector<float> &embedding) {
    sqlite3_bind_blob(stmt, index, embedding.data(), static_cast<int>(embedding.size() * sizeof(float)), SQLITE_TRANSIENT);
}

} // namespace

TEST_CASE("libSQL-backed corpus store persists vectors and exact cosine search", "[rag][store][libsql]") {
    CorpusConfig config;
    config.storage_path = unique_db_path("store.db");

    std::unique_ptr<CorpusStore> store;
    REQUIRE_FALSE(make_libsql_corpus_store(config, store));

    REQUIRE_FALSE(store->set_embedding_state({"fixture", 2, true, VectorMetric::Cosine, 0}));

    const SourceRecord source_alpha = make_source("doc-alpha", "alpha text", {{"topic", "alpha"}});
    const SourceRecord source_beta = make_source("doc-beta", "beta text", {{"topic", "beta"}});

    IngestStats stats;
    REQUIRE_FALSE(store->upsert_document(source_alpha,
                                         {make_chunk(source_alpha, 0, 0, 10, "alpha chunk", {1.0f, 0.0f})},
                                         stats));
    REQUIRE(stats.chunks_written == 1);
    REQUIRE_FALSE(store->upsert_document(source_beta,
                                         {make_chunk(source_beta, 0, 0, 10, "beta chunk", {0.0f, 1.0f})},
                                         stats));

    std::vector<VectorSearchHit> hits;
    RetrievalOptions options;
    options.candidate_k = 8;
    REQUIRE_FALSE(store->exact_vector_search({1.0f, 0.0f}, options, hits));
    REQUIRE(hits.size() == 2);

    std::vector<ChunkRecord> chunks;
    REQUIRE_FALSE(store->fetch_chunks_by_ids({hits[0].chunk_id, hits[1].chunk_id}, true, chunks));
    REQUIRE(chunks.size() == 2);
    REQUIRE(chunks[0].source_id == "doc-alpha");
    REQUIRE(chunks[0].embedding == std::vector<float>({1.0f, 0.0f}));

    options.metadata_filter = {{"topic", "beta"}};
    hits.clear();
    REQUIRE_FALSE(store->exact_vector_search({0.0f, 1.0f}, options, hits));
    REQUIRE(hits.size() == 1);
    REQUIRE_FALSE(store->fetch_chunks_by_ids({hits[0].chunk_id}, false, chunks));
    REQUIRE(chunks.size() == 1);
    REQUIRE(chunks[0].source_id == "doc-beta");
    REQUIRE(chunks[0].embedding.empty());
}

TEST_CASE("libSQL-backed corpus store migrates schema v1 embeddings to schema v2", "[rag][store][migration]") {
    const auto path = unique_db_path("migration.db");

    sqlite3 *db = nullptr;
    REQUIRE(sqlite3_open(path.string().c_str(), &db) == SQLITE_OK);

    exec_sql(db, R"SQL(
CREATE TABLE rag_meta(
    key_name TEXT PRIMARY KEY,
    value_text TEXT NOT NULL
);
CREATE TABLE sources(
    source_id TEXT PRIMARY KEY,
    source_version TEXT NOT NULL,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    metadata_blob TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE source_metadata(
    source_id TEXT NOT NULL,
    key_name TEXT NOT NULL,
    value_text TEXT NOT NULL,
    PRIMARY KEY(source_id, key_name)
);
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
    embedding_blob BLOB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_chunks_source ON chunks(source_id, chunk_index);
CREATE INDEX idx_source_metadata_key_value ON source_metadata(key_name, value_text);
INSERT INTO rag_meta(key_name, value_text) VALUES
    ('schema_version', '1'),
    ('embedding_fingerprint', 'fixture'),
    ('embedding_dimensions', '2'),
    ('embedding_normalized', '1'),
    ('vector_metric', 'cosine'),
    ('pooling_type', '0');
INSERT INTO sources(source_id, source_version, title, source_path, normalized_text, metadata_blob, created_at, updated_at)
VALUES('doc-old', 'v1', 'doc-old', 'doc-old.txt', 'old text', 'topic\tlegacy\n', '2026-04-10T00:00:00Z', '2026-04-10T00:00:00Z');
INSERT INTO source_metadata(source_id, key_name, value_text) VALUES('doc-old', 'topic', 'legacy');
)SQL");

    const std::vector<float> embedding = {1.0f, 0.0f};
    sqlite3_stmt *stmt = nullptr;
    REQUIRE(sqlite3_prepare_v2(
                    db,
                    "INSERT INTO chunks(chunk_id, source_id, source_version, title, source_path, normalized_text, "
                    "display_text, metadata_blob, chunk_index, byte_start, byte_end, char_start, char_end, token_count, "
                    "embedding_fingerprint, embedding_dimensions, embedding_normalized, vector_metric, pooling_type, "
                    "embedding_blob, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    -1, &stmt, nullptr) == SQLITE_OK);
    bind_text(stmt, 1, "chunk-old");
    bind_text(stmt, 2, "doc-old");
    bind_text(stmt, 3, "v1");
    bind_text(stmt, 4, "doc-old");
    bind_text(stmt, 5, "doc-old.txt");
    bind_text(stmt, 6, "old text");
    bind_text(stmt, 7, "old chunk");
    bind_text(stmt, 8, "topic\tlegacy\n");
    sqlite3_bind_int(stmt, 9, 0);
    sqlite3_bind_int64(stmt, 10, 0);
    sqlite3_bind_int64(stmt, 11, 8);
    sqlite3_bind_int(stmt, 12, 0);
    sqlite3_bind_int(stmt, 13, 8);
    sqlite3_bind_int(stmt, 14, 4);
    bind_text(stmt, 15, "fixture");
    sqlite3_bind_int(stmt, 16, 2);
    sqlite3_bind_int(stmt, 17, 1);
    bind_text(stmt, 18, "cosine");
    sqlite3_bind_int(stmt, 19, 0);
    bind_blob(stmt, 20, embedding);
    bind_text(stmt, 21, "2026-04-10T00:00:00Z");
    bind_text(stmt, 22, "2026-04-10T00:00:00Z");
    REQUIRE(sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    CorpusConfig config;
    config.storage_path = path;

    std::unique_ptr<CorpusStore> store;
    REQUIRE_FALSE(make_libsql_corpus_store(config, store));

    CorpusStats corpus_stats;
    REQUIRE_FALSE(store->get_stats(corpus_stats));
    REQUIRE(corpus_stats.schema_version == 2);
    REQUIRE(corpus_stats.embedding_dimensions == 2);

    std::vector<VectorSearchHit> hits;
    RetrievalOptions options;
    options.candidate_k = 4;
    options.metadata_filter = {{"topic", "legacy"}};
    REQUIRE_FALSE(store->exact_vector_search({1.0f, 0.0f}, options, hits));
    REQUIRE(hits.size() == 1);

    std::vector<ChunkRecord> chunks;
    REQUIRE_FALSE(store->fetch_chunks_by_ids({hits[0].chunk_id}, true, chunks));
    REQUIRE(chunks.size() == 1);
    REQUIRE(chunks[0].source_id == "doc-old");
    REQUIRE(chunks[0].embedding == embedding);
}
