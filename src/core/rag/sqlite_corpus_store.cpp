#include "godot_llama/rag/factories.hpp"

#include <sqlite3.h>

#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace godot_llama::rag {
namespace {

constexpr int32_t kSchemaVersion = 1;

std::string serialize_metadata(const Metadata &metadata) {
    std::string serialized;
    for (const auto &entry : metadata) {
        serialized.append(entry.key);
        serialized.push_back('\t');
        serialized.append(entry.value);
        serialized.push_back('\n');
    }
    return serialized;
}

Metadata deserialize_metadata(std::string_view serialized) {
    Metadata metadata;
    size_t cursor = 0;
    while (cursor < serialized.size()) {
        size_t line_end = serialized.find('\n', cursor);
        if (line_end == std::string_view::npos) {
            line_end = serialized.size();
        }
        const std::string_view line = serialized.substr(cursor, line_end - cursor);
        if (!line.empty()) {
            const size_t tab = line.find('\t');
            if (tab != std::string_view::npos) {
                metadata.push_back({std::string(line.substr(0, tab)), std::string(line.substr(tab + 1))});
            }
        }
        cursor = line_end + 1;
    }
    return metadata;
}

const char *vector_metric_sql(VectorMetric metric) {
    return metric == VectorMetric::Dot ? "dot" : "cosine";
}

Error sqlite_error(sqlite3 *db, ErrorCode code, std::string_view message) {
    std::string detail(message);
    if (db) {
        detail.append(": ");
        detail.append(sqlite3_errmsg(db));
    }
    return Error::make(code, detail);
}

class Statement {
public:
    Statement(sqlite3 *db, const char *sql) : db_(db) {
        sqlite3_prepare_v2(db_, sql, -1, &stmt_, nullptr);
    }

    ~Statement() {
        if (stmt_) {
            sqlite3_finalize(stmt_);
        }
    }

    Statement(const Statement &) = delete;
    Statement &operator=(const Statement &) = delete;

    [[nodiscard]] bool valid() const noexcept { return stmt_ != nullptr; }
    [[nodiscard]] sqlite3_stmt *get() const noexcept { return stmt_; }

private:
    sqlite3 *db_ = nullptr;
    sqlite3_stmt *stmt_ = nullptr;
};

class SqliteCorpusStore final : public CorpusStore {
public:
    explicit SqliteCorpusStore(std::filesystem::path database_path) : database_path_(std::move(database_path)) {}
    ~SqliteCorpusStore() override { close(); }

    [[nodiscard]] Error open() {
        if (db_) {
            return Error::make_ok();
        }

        std::error_code ec;
        const auto parent = database_path_.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                return Error::make(ErrorCode::StorageOpenFailed, "Failed to create storage directory", parent.string());
            }
        }

        if (sqlite3_open(database_path_.string().c_str(), &db_) != SQLITE_OK) {
            return sqlite_error(db_, ErrorCode::StorageOpenFailed, "Failed to open SQLite corpus store");
        }

        return initialize_schema();
    }

    [[nodiscard]] bool is_open() const noexcept override { return db_ != nullptr; }

    void close() noexcept override {
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }

    [[nodiscard]] Error set_embedding_state(const EmbeddingInfo &embedding_info) override {
        return with_transaction([&]() -> Error {
            Error err = put_meta("embedding_fingerprint", embedding_info.model_fingerprint);
            if (err) {
                return err;
            }
            err = put_meta("embedding_dimensions", std::to_string(embedding_info.dimensions));
            if (err) {
                return err;
            }
            err = put_meta("embedding_normalized", embedding_info.normalized ? "1" : "0");
            if (err) {
                return err;
            }
            err = put_meta("vector_metric", vector_metric_sql(embedding_info.metric));
            if (err) {
                return err;
            }
            return put_meta("pooling_type", std::to_string(embedding_info.pooling_type));
        });
    }

    [[nodiscard]] Error get_embedding_state(EmbeddingInfo &out_info, bool &out_present) const override {
        out_info = {};
        out_present = false;

        std::string fingerprint;
        Error err = get_meta("embedding_fingerprint", fingerprint, out_present);
        if (err || !out_present) {
            return err;
        }

        std::string dimensions;
        bool present = false;
        err = get_meta("embedding_dimensions", dimensions, present);
        if (err || !present) {
            return err ? err : Error::make(ErrorCode::StorageCorrupt, "Missing embedding_dimensions metadata");
        }

        std::string normalized;
        err = get_meta("embedding_normalized", normalized, present);
        if (err || !present) {
            return err ? err : Error::make(ErrorCode::StorageCorrupt, "Missing embedding_normalized metadata");
        }

        std::string metric;
        err = get_meta("vector_metric", metric, present);
        if (err || !present) {
            return err ? err : Error::make(ErrorCode::StorageCorrupt, "Missing vector_metric metadata");
        }

        std::string pooling_type;
        err = get_meta("pooling_type", pooling_type, present);
        if (err || !present) {
            return err ? err : Error::make(ErrorCode::StorageCorrupt, "Missing pooling_type metadata");
        }

        out_info.model_fingerprint = std::move(fingerprint);
        out_info.dimensions = std::stoi(dimensions);
        out_info.normalized = normalized == "1";
        out_info.metric = parse_vector_metric(metric).value_or(VectorMetric::Cosine);
        out_info.pooling_type = std::stoi(pooling_type);
        out_present = true;
        return Error::make_ok();
    }

    [[nodiscard]] Error upsert_document(const SourceRecord &source, const std::vector<ChunkRecord> &chunks,
                                        IngestStats &out_stats) override {
        int32_t previous_chunk_count = 0;
        Error err = count_source_chunks(source.source_id, previous_chunk_count);
        if (err) {
            return err;
        }

        err = with_transaction([&]() -> Error {
            Statement source_stmt(db_,
                                  "INSERT INTO sources(source_id, source_version, title, source_path, normalized_text, "
                                  "metadata_blob, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?) "
                                  "ON CONFLICT(source_id) DO UPDATE SET source_version=excluded.source_version, "
                                  "title=excluded.title, source_path=excluded.source_path, normalized_text=excluded.normalized_text, "
                                  "metadata_blob=excluded.metadata_blob, updated_at=excluded.updated_at");
            if (!source_stmt.valid()) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare source upsert");
            }

            bind_text(source_stmt.get(), 1, source.source_id);
            bind_text(source_stmt.get(), 2, source.source_version);
            bind_text(source_stmt.get(), 3, source.title);
            bind_text(source_stmt.get(), 4, source.source_path);
            bind_text(source_stmt.get(), 5, source.normalized_text);
            bind_text(source_stmt.get(), 6, serialize_metadata(source.metadata));
            bind_text(source_stmt.get(), 7, source.created_at);
            bind_text(source_stmt.get(), 8, source.updated_at);
            if (sqlite3_step(source_stmt.get()) != SQLITE_DONE) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to upsert source row");
            }

            Statement delete_source_metadata(db_, "DELETE FROM source_metadata WHERE source_id = ?");
            if (!delete_source_metadata.valid()) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare source metadata deletion");
            }
            bind_text(delete_source_metadata.get(), 1, source.source_id);
            if (sqlite3_step(delete_source_metadata.get()) != SQLITE_DONE) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to clear old source metadata");
            }

            Statement source_metadata_stmt(db_,
                                           "INSERT INTO source_metadata(source_id, key_name, value_text) VALUES(?, ?, ?)");
            if (!source_metadata_stmt.valid()) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare source metadata upsert");
            }

            for (const auto &entry : source.metadata) {
                sqlite3_reset(source_metadata_stmt.get());
                sqlite3_clear_bindings(source_metadata_stmt.get());
                bind_text(source_metadata_stmt.get(), 1, source.source_id);
                bind_text(source_metadata_stmt.get(), 2, entry.key);
                bind_text(source_metadata_stmt.get(), 3, entry.value);
                if (sqlite3_step(source_metadata_stmt.get()) != SQLITE_DONE) {
                    return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to insert source metadata row");
                }
            }

            Statement delete_chunks(db_, "DELETE FROM chunks WHERE source_id = ?");
            if (!delete_chunks.valid()) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare chunk deletion");
            }
            bind_text(delete_chunks.get(), 1, source.source_id);
            if (sqlite3_step(delete_chunks.get()) != SQLITE_DONE) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to delete previous chunks");
            }

            Statement chunk_stmt(
                    db_,
                    "INSERT INTO chunks(chunk_id, source_id, source_version, title, source_path, normalized_text, "
                    "display_text, metadata_blob, chunk_index, byte_start, byte_end, char_start, char_end, token_count, "
                    "embedding_fingerprint, embedding_dimensions, embedding_normalized, vector_metric, pooling_type, "
                    "embedding_blob, created_at, updated_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
            if (!chunk_stmt.valid()) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare chunk insert");
            }

            const std::string timestamp = utc_timestamp_now();
            for (const auto &chunk : chunks) {
                sqlite3_reset(chunk_stmt.get());
                sqlite3_clear_bindings(chunk_stmt.get());
                bind_text(chunk_stmt.get(), 1, chunk.chunk_id);
                bind_text(chunk_stmt.get(), 2, chunk.source_id);
                bind_text(chunk_stmt.get(), 3, chunk.source_version);
                bind_text(chunk_stmt.get(), 4, chunk.title);
                bind_text(chunk_stmt.get(), 5, chunk.source_path);
                bind_text(chunk_stmt.get(), 6, chunk.normalized_text);
                bind_text(chunk_stmt.get(), 7, chunk.display_text);
                bind_text(chunk_stmt.get(), 8, serialize_metadata(chunk.metadata));
                sqlite3_bind_int(chunk_stmt.get(), 9, chunk.chunk_index);
                sqlite3_bind_int64(chunk_stmt.get(), 10, chunk.byte_start);
                sqlite3_bind_int64(chunk_stmt.get(), 11, chunk.byte_end);
                sqlite3_bind_int(chunk_stmt.get(), 12, chunk.char_start);
                sqlite3_bind_int(chunk_stmt.get(), 13, chunk.char_end);
                sqlite3_bind_int(chunk_stmt.get(), 14, chunk.token_count);
                bind_text(chunk_stmt.get(), 15, chunk.embedding_info.model_fingerprint);
                sqlite3_bind_int(chunk_stmt.get(), 16, chunk.embedding_info.dimensions);
                sqlite3_bind_int(chunk_stmt.get(), 17, chunk.embedding_info.normalized ? 1 : 0);
                bind_text(chunk_stmt.get(), 18, vector_metric_sql(chunk.embedding_info.metric));
                sqlite3_bind_int(chunk_stmt.get(), 19, chunk.embedding_info.pooling_type);

                const int blob_size = static_cast<int>(chunk.embedding.size() * sizeof(float));
                sqlite3_bind_blob(chunk_stmt.get(), 20, chunk.embedding.data(), blob_size, SQLITE_TRANSIENT);
                bind_text(chunk_stmt.get(), 21, timestamp);
                bind_text(chunk_stmt.get(), 22, timestamp);
                if (sqlite3_step(chunk_stmt.get()) != SQLITE_DONE) {
                    return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to insert chunk row");
                }
            }

            return Error::make_ok();
        });

        if (err) {
            return err;
        }

        out_stats = {};
        out_stats.source_id = source.source_id;
        out_stats.source_version = source.source_version;
        out_stats.chunks_deleted = previous_chunk_count;
        out_stats.chunks_written = static_cast<int32_t>(chunks.size());
        out_stats.embeddings_generated = static_cast<int32_t>(chunks.size());
        return Error::make_ok();
    }

    [[nodiscard]] Error delete_source(std::string_view source_id, IngestStats &out_stats) override {
        int32_t chunk_count = 0;
        Error err = count_source_chunks(source_id, chunk_count);
        if (err) {
            return err;
        }

        Statement stmt(db_, "DELETE FROM sources WHERE source_id = ?");
        if (!stmt.valid()) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare source deletion");
        }
        bind_text(stmt.get(), 1, source_id);
        if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to delete source row");
        }

        out_stats = {};
        out_stats.source_id = std::string(source_id);
        out_stats.chunks_deleted = chunk_count;
        return Error::make_ok();
    }

    [[nodiscard]] Error clear(IngestStats &out_stats) override {
        Error err = with_transaction([&]() -> Error {
            Statement delete_chunks(db_, "DELETE FROM chunks");
            Statement delete_source_metadata(db_, "DELETE FROM source_metadata");
            Statement delete_sources(db_, "DELETE FROM sources");
            if (!delete_chunks.valid() || !delete_source_metadata.valid() || !delete_sources.valid()) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare clear statements");
            }
            if (sqlite3_step(delete_chunks.get()) != SQLITE_DONE || sqlite3_step(delete_source_metadata.get()) != SQLITE_DONE ||
                sqlite3_step(delete_sources.get()) != SQLITE_DONE) {
                return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to clear corpus tables");
            }
            return Error::make_ok();
        });
        if (!err) {
            out_stats = {};
        }
        return err;
    }

    [[nodiscard]] Error list_sources(std::vector<SourceRecord> &out_sources) const override {
        out_sources.clear();
        Statement stmt(db_,
                       "SELECT source_id, source_version, title, source_path, normalized_text, metadata_blob, created_at, "
                       "updated_at FROM sources ORDER BY source_id");
        if (!stmt.valid()) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare source listing");
        }

        while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
            SourceRecord source;
            source.source_id = column_text(stmt.get(), 0);
            source.source_version = column_text(stmt.get(), 1);
            source.title = column_text(stmt.get(), 2);
            source.source_path = column_text(stmt.get(), 3);
            source.normalized_text = column_text(stmt.get(), 4);
            source.metadata = deserialize_metadata(column_text(stmt.get(), 5));
            source.created_at = column_text(stmt.get(), 6);
            source.updated_at = column_text(stmt.get(), 7);
            out_sources.push_back(std::move(source));
        }
        return Error::make_ok();
    }

    [[nodiscard]] Error get_source(std::string_view source_id, std::optional<SourceRecord> &out_source) const override {
        out_source.reset();
        Statement stmt(db_,
                       "SELECT source_id, source_version, title, source_path, normalized_text, metadata_blob, created_at, "
                       "updated_at FROM sources WHERE source_id = ?");
        if (!stmt.valid()) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare source fetch");
        }
        bind_text(stmt.get(), 1, source_id);
        if (sqlite3_step(stmt.get()) != SQLITE_ROW) {
            return Error::make_ok();
        }

        SourceRecord source;
        source.source_id = column_text(stmt.get(), 0);
        source.source_version = column_text(stmt.get(), 1);
        source.title = column_text(stmt.get(), 2);
        source.source_path = column_text(stmt.get(), 3);
        source.normalized_text = column_text(stmt.get(), 4);
        source.metadata = deserialize_metadata(column_text(stmt.get(), 5));
        source.created_at = column_text(stmt.get(), 6);
        source.updated_at = column_text(stmt.get(), 7);
        out_source = std::move(source);
        return Error::make_ok();
    }

    [[nodiscard]] Error fetch_candidate_chunks(const RetrievalOptions &options,
                                               std::vector<ChunkRecord> &out_chunks) const override {
        out_chunks.clear();

        std::string sql =
                "SELECT chunk_id, source_id, source_version, title, source_path, normalized_text, display_text, "
                "metadata_blob, chunk_index, byte_start, byte_end, char_start, char_end, token_count, "
                "embedding_fingerprint, embedding_dimensions, embedding_normalized, vector_metric, pooling_type, "
                "embedding_blob FROM chunks c WHERE 1=1";

        const int source_id_count = static_cast<int>(options.source_ids.size());
        if (source_id_count > 0) {
            sql.append(" AND c.source_id IN (");
            for (int index = 0; index < source_id_count; ++index) {
                sql.append(index == 0 ? "?" : ", ?");
            }
            sql.push_back(')');
        }

        const int exclude_count = static_cast<int>(options.exclude_source_ids.size());
        if (exclude_count > 0) {
            sql.append(" AND c.source_id NOT IN (");
            for (int index = 0; index < exclude_count; ++index) {
                sql.append(index == 0 ? "?" : ", ?");
            }
            sql.push_back(')');
        }

        for (size_t index = 0; index < options.metadata_filter.size(); ++index) {
            sql.append(" AND EXISTS (SELECT 1 FROM source_metadata sm");
            sql.append(std::to_string(index));
            sql.append(" WHERE sm");
            sql.append(std::to_string(index));
            sql.append(".source_id = c.source_id AND sm");
            sql.append(std::to_string(index));
            sql.append(".key_name = ? AND sm");
            sql.append(std::to_string(index));
            sql.append(".value_text = ?)");
        }

        sql.append(" ORDER BY c.source_id, c.chunk_index");
        Statement stmt(db_, sql.c_str());
        if (!stmt.valid()) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare candidate chunk query");
        }

        int bind_index = 1;
        for (const auto &source_id : options.source_ids) {
            bind_text(stmt.get(), bind_index++, source_id);
        }
        for (const auto &source_id : options.exclude_source_ids) {
            bind_text(stmt.get(), bind_index++, source_id);
        }
        for (const auto &entry : options.metadata_filter) {
            bind_text(stmt.get(), bind_index++, entry.key);
            bind_text(stmt.get(), bind_index++, entry.value);
        }

        while (sqlite3_step(stmt.get()) == SQLITE_ROW) {
            ChunkRecord chunk;
            chunk.chunk_id = column_text(stmt.get(), 0);
            chunk.source_id = column_text(stmt.get(), 1);
            chunk.source_version = column_text(stmt.get(), 2);
            chunk.title = column_text(stmt.get(), 3);
            chunk.source_path = column_text(stmt.get(), 4);
            chunk.normalized_text = column_text(stmt.get(), 5);
            chunk.display_text = column_text(stmt.get(), 6);
            chunk.metadata = deserialize_metadata(column_text(stmt.get(), 7));
            chunk.chunk_index = sqlite3_column_int(stmt.get(), 8);
            chunk.byte_start = sqlite3_column_int64(stmt.get(), 9);
            chunk.byte_end = sqlite3_column_int64(stmt.get(), 10);
            chunk.char_start = sqlite3_column_int(stmt.get(), 11);
            chunk.char_end = sqlite3_column_int(stmt.get(), 12);
            chunk.token_count = sqlite3_column_int(stmt.get(), 13);
            chunk.embedding_info.model_fingerprint = column_text(stmt.get(), 14);
            chunk.embedding_info.dimensions = sqlite3_column_int(stmt.get(), 15);
            chunk.embedding_info.normalized = sqlite3_column_int(stmt.get(), 16) != 0;
            chunk.embedding_info.metric = parse_vector_metric(column_text(stmt.get(), 17)).value_or(VectorMetric::Cosine);
            chunk.embedding_info.pooling_type = sqlite3_column_int(stmt.get(), 18);

            const void *blob = sqlite3_column_blob(stmt.get(), 19);
            const int blob_size = sqlite3_column_bytes(stmt.get(), 19);
            if (blob_size % static_cast<int>(sizeof(float)) != 0) {
                return Error::make(ErrorCode::StorageCorrupt, "Invalid embedding blob size", chunk.chunk_id);
            }
            const size_t element_count = static_cast<size_t>(blob_size / static_cast<int>(sizeof(float)));
            chunk.embedding.resize(element_count);
            if (blob_size > 0) {
                memcpy(chunk.embedding.data(), blob, static_cast<size_t>(blob_size));
            }
            out_chunks.push_back(std::move(chunk));
        }

        return Error::make_ok();
    }

    [[nodiscard]] Error get_stats(CorpusStats &out_stats) const override {
        out_stats = {};

        Statement stmt(db_, "SELECT (SELECT COUNT(*) FROM sources), (SELECT COUNT(*) FROM chunks)");
        if (!stmt.valid()) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare corpus stats query");
        }
        if (sqlite3_step(stmt.get()) == SQLITE_ROW) {
            out_stats.source_count = sqlite3_column_int64(stmt.get(), 0);
            out_stats.chunk_count = sqlite3_column_int64(stmt.get(), 1);
        }

        std::string schema_version;
        bool present = false;
        Error err = get_meta("schema_version", schema_version, present);
        if (err) {
            return err;
        }
        out_stats.schema_version = present ? std::stoi(schema_version) : 0;

        EmbeddingInfo embedding_info;
        err = get_embedding_state(embedding_info, present);
        if (err) {
            return err;
        }
        if (present) {
            out_stats.embedding_model_fingerprint = embedding_info.model_fingerprint;
            out_stats.embedding_dimensions = embedding_info.dimensions;
            out_stats.embedding_normalized = embedding_info.normalized;
            out_stats.vector_metric = embedding_info.metric;
        }

        return Error::make_ok();
    }

private:
    [[nodiscard]] Error initialize_schema() {
        const char *schema_sql = R"SQL(
CREATE TABLE IF NOT EXISTS rag_meta(
    key_name TEXT PRIMARY KEY,
    value_text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources(
    source_id TEXT PRIMARY KEY,
    source_version TEXT NOT NULL,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    metadata_blob TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_metadata(
    source_id TEXT NOT NULL,
    key_name TEXT NOT NULL,
    value_text TEXT NOT NULL,
    PRIMARY KEY(source_id, key_name),
    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS chunks(
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
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_source_metadata_key_value ON source_metadata(key_name, value_text);
)SQL";

        char *error_message = nullptr;
        if (sqlite3_exec(db_, "PRAGMA foreign_keys = ON", nullptr, nullptr, &error_message) != SQLITE_OK) {
            const std::string detail = error_message ? error_message : "unknown SQLite error";
            sqlite3_free(error_message);
            return Error::make(ErrorCode::StorageOpenFailed, "Failed to enable foreign keys", detail);
        }
        if (sqlite3_exec(db_, schema_sql, nullptr, nullptr, &error_message) != SQLITE_OK) {
            const std::string detail = error_message ? error_message : "unknown SQLite error";
            sqlite3_free(error_message);
            return Error::make(ErrorCode::StorageMigrationFailed, "Failed to initialize corpus schema", detail);
        }

        return put_meta("schema_version", std::to_string(kSchemaVersion));
    }

    [[nodiscard]] Error put_meta(std::string_view key, std::string_view value) const {
        Statement stmt(
                db_,
                "INSERT INTO rag_meta(key_name, value_text) VALUES(?, ?) ON CONFLICT(key_name) DO UPDATE SET value_text=excluded.value_text");
        if (!stmt.valid()) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare rag_meta upsert");
        }
        bind_text(stmt.get(), 1, key);
        bind_text(stmt.get(), 2, value);
        if (sqlite3_step(stmt.get()) != SQLITE_DONE) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to write rag_meta value");
        }
        return Error::make_ok();
    }

    [[nodiscard]] Error get_meta(std::string_view key, std::string &out_value, bool &out_present) const {
        out_value.clear();
        out_present = false;
        Statement stmt(db_, "SELECT value_text FROM rag_meta WHERE key_name = ?");
        if (!stmt.valid()) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare rag_meta fetch");
        }
        bind_text(stmt.get(), 1, key);
        if (sqlite3_step(stmt.get()) == SQLITE_ROW) {
            out_value = column_text(stmt.get(), 0);
            out_present = true;
        }
        return Error::make_ok();
    }

    [[nodiscard]] Error with_transaction(const std::function<Error()> &fn) const {
        char *error_message = nullptr;
        if (sqlite3_exec(db_, "BEGIN IMMEDIATE", nullptr, nullptr, &error_message) != SQLITE_OK) {
            const std::string detail = error_message ? error_message : "unknown SQLite error";
            sqlite3_free(error_message);
            return Error::make(ErrorCode::StorageCorrupt, "Failed to begin transaction", detail);
        }

        Error err = fn();
        if (err) {
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
            return err;
        }

        if (sqlite3_exec(db_, "COMMIT", nullptr, nullptr, &error_message) != SQLITE_OK) {
            const std::string detail = error_message ? error_message : "unknown SQLite error";
            sqlite3_free(error_message);
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
            return Error::make(ErrorCode::StorageCorrupt, "Failed to commit transaction", detail);
        }

        return Error::make_ok();
    }

    static void bind_text(sqlite3_stmt *stmt, int index, std::string_view value) {
        sqlite3_bind_text(stmt, index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
    }

    static std::string column_text(sqlite3_stmt *stmt, int index) {
        const unsigned char *value = sqlite3_column_text(stmt, index);
        if (!value) {
            return {};
        }
        return reinterpret_cast<const char *>(value);
    }

    [[nodiscard]] Error count_source_chunks(std::string_view source_id, int32_t &out_count) const {
        out_count = 0;
        Statement stmt(db_, "SELECT COUNT(*) FROM chunks WHERE source_id = ?");
        if (!stmt.valid()) {
            return sqlite_error(db_, ErrorCode::StorageCorrupt, "Failed to prepare source chunk count");
        }
        bind_text(stmt.get(), 1, source_id);
        if (sqlite3_step(stmt.get()) == SQLITE_ROW) {
            out_count = sqlite3_column_int(stmt.get(), 0);
        }
        return Error::make_ok();
    }

    sqlite3 *db_ = nullptr;
    std::filesystem::path database_path_;
};

std::filesystem::path resolve_database_path(const std::filesystem::path &storage_path) {
    if (storage_path.empty()) {
        return {};
    }
    std::filesystem::path normalized = storage_path.lexically_normal();
    if (!normalized.has_extension()) {
        normalized /= "rag_corpus.sqlite3";
    }
    return normalized;
}

} // namespace

Error make_sqlite_corpus_store(const CorpusConfig &config, std::unique_ptr<CorpusStore> &out_store) {
    out_store.reset();
    const std::filesystem::path database_path = resolve_database_path(config.storage_path);
    if (database_path.empty()) {
        return Error::make(ErrorCode::InvalidPath, "storage_path is empty");
    }

    auto store = std::make_unique<SqliteCorpusStore>(database_path);
    Error err = store->open();
    if (err) {
        return err;
    }

    out_store = std::move(store);
    return Error::make_ok();
}

} // namespace godot_llama::rag
