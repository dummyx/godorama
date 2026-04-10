#pragma once

#include "godot_llama/error.hpp"
#include "godot_llama/llama_params.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace godot_llama::rag {

using CancelCheck = std::function<bool()>;
using ProgressCallback = std::function<void(int32_t, int32_t)>;

struct MetadataEntry {
    std::string key;
    std::string value;
};

using Metadata = std::vector<MetadataEntry>;

enum class VectorMetric : int32_t {
    Cosine = 0,
    Dot = 1,
};

enum class ParserMode : int32_t {
    Auto = 0,
    Text = 1,
    Markdown = 2,
};

struct ChunkingConfig {
    int32_t chunk_size_tokens = 256;
    int32_t chunk_overlap_tokens = 32;
};

struct CorpusConfig {
    std::filesystem::path storage_path;
    ChunkingConfig chunking;
    bool normalize_embeddings = true;
    VectorMetric vector_metric = VectorMetric::Cosine;
    int32_t max_batch_texts = 8;
    ModelConfig embedding_model;
    bool enable_reranker = false;
    ModelConfig reranker_model;
    ParserMode parser_mode = ParserMode::Auto;
    std::vector<std::string> supported_extensions = {".txt", ".md", ".markdown"};
    int32_t max_metadata_entries = 32;
    int32_t max_metadata_key_bytes = 64;
    int32_t max_metadata_value_bytes = 256;
    int32_t max_queue_depth = 32;
};

struct NormalizedDocument {
    std::string source_id;
    std::string source_version;
    std::string title;
    std::string source_path;
    std::string normalized_text;
    Metadata metadata;
    ParserMode parser_mode = ParserMode::Text;
    int32_t char_count = 0;
};

struct EmbeddingInfo {
    std::string model_fingerprint;
    int32_t dimensions = 0;
    bool normalized = false;
    VectorMetric metric = VectorMetric::Cosine;
    int32_t pooling_type = 0;
};

struct SourceRecord {
    std::string source_id;
    std::string source_version;
    std::string title;
    std::string source_path;
    std::string normalized_text;
    Metadata metadata;
    std::string created_at;
    std::string updated_at;
};

struct ChunkRecord {
    std::string chunk_id;
    std::string source_id;
    std::string source_version;
    std::string title;
    std::string source_path;
    std::string normalized_text;
    std::string display_text;
    Metadata metadata;
    int32_t chunk_index = 0;
    int64_t byte_start = 0;
    int64_t byte_end = 0;
    int32_t char_start = 0;
    int32_t char_end = 0;
    int32_t token_count = 0;
    EmbeddingInfo embedding_info;
    std::vector<float> embedding;
};

struct RetrievalOptions {
    int32_t top_k = 5;
    int32_t candidate_k = 20;
    float score_threshold = 0.0f;
    int32_t max_context_chunks = 4;
    int32_t max_context_tokens = 1200;
    Metadata metadata_filter;
    std::vector<std::string> source_ids;
    std::vector<std::string> exclude_source_ids;
    bool use_mmr = true;
    bool use_reranker = false;
    float mmr_lambda = 0.75f;
};

struct RetrievalHit {
    std::string chunk_id;
    std::string source_id;
    std::string title;
    std::string source_path;
    std::string excerpt;
    Metadata metadata;
    float raw_score = 0.0f;
    float relevance_score = 0.0f;
    std::optional<float> rerank_score;
    int64_t byte_start = 0;
    int64_t byte_end = 0;
    int32_t char_start = 0;
    int32_t char_end = 0;
    int32_t token_count = 0;
};

struct RetrievalStats {
    int32_t query_token_count = 0;
    int32_t scanned_chunks = 0;
    int32_t candidate_chunks = 0;
    int32_t filtered_chunks = 0;
    int32_t deduplicated_chunks = 0;
    int32_t returned_chunks = 0;
    bool ann_fallback_used = false;
    bool reranker_used = false;
    std::string search_mode;
    std::string reranker_status;
    std::vector<std::string> truncated_chunk_ids;
};

struct Citation {
    std::string chunk_id;
    std::string source_id;
    std::string title;
    std::string source_path;
    int64_t byte_start = 0;
    int64_t byte_end = 0;
    int32_t char_start = 0;
    int32_t char_end = 0;
    std::string excerpt;
};

struct PromptChunk {
    RetrievalHit hit;
    std::string prompt_text;
    int32_t prompt_token_count = 0;
};

struct PromptAssembly {
    std::string prompt;
    std::vector<PromptChunk> packed_chunks;
    std::vector<Citation> citations;
    int32_t prompt_token_count = 0;
    int32_t context_token_count = 0;
    int32_t truncated_chunks = 0;
    bool abstained = false;
    std::string prompt_style;
};

struct CorpusStats {
    int32_t schema_version = 0;
    int64_t source_count = 0;
    int64_t chunk_count = 0;
    std::string embedding_model_fingerprint;
    int32_t embedding_dimensions = 0;
    bool embedding_normalized = false;
    VectorMetric vector_metric = VectorMetric::Cosine;
    bool stale_embeddings = false;
};

struct IngestStats {
    std::string source_id;
    std::string source_version;
    int32_t chunks_written = 0;
    int32_t chunks_reused = 0;
    int32_t chunks_deleted = 0;
    int32_t embeddings_generated = 0;
};

struct AnswerStats {
    RetrievalStats retrieval;
    int32_t packed_chunks = 0;
    int32_t packed_context_tokens = 0;
    int32_t truncated_chunks = 0;
    bool abstained = false;
    std::string prompt_style;
};

[[nodiscard]] const char *vector_metric_name(VectorMetric metric) noexcept;
[[nodiscard]] std::optional<VectorMetric> parse_vector_metric(std::string_view value) noexcept;

[[nodiscard]] const char *parser_mode_name(ParserMode mode) noexcept;
[[nodiscard]] std::optional<ParserMode> parse_parser_mode(std::string_view value) noexcept;

[[nodiscard]] std::string canonicalize_metadata_value(std::string_view value);
[[nodiscard]] Metadata canonicalize_metadata(Metadata metadata);
[[nodiscard]] std::optional<std::string> metadata_lookup(const Metadata &metadata, std::string_view key);
[[nodiscard]] bool metadata_matches(const Metadata &metadata, const Metadata &filter);

[[nodiscard]] std::string stable_hash_hex(std::string_view value) noexcept;
[[nodiscard]] std::string make_source_version(std::string_view normalized_text) noexcept;
[[nodiscard]] std::string make_chunk_id(std::string_view source_id, std::string_view source_version, int64_t byte_start,
                                        int64_t byte_end, int32_t chunk_index) noexcept;
[[nodiscard]] std::string utc_timestamp_now();

} // namespace godot_llama::rag
