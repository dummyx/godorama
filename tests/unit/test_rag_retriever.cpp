#include <catch2/catch_test_macros.hpp>

#include "godot_llama/rag/factories.hpp"
#include "godot_llama/rag/mock_embedder.hpp"

#include <algorithm>
#include <cmath>

using namespace godot_llama::rag;

namespace {

float cosine_distance(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    const size_t size = std::min(lhs.size(), rhs.size());
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (size_t index = 0; index < size; ++index) {
        dot += static_cast<double>(lhs[index]) * static_cast<double>(rhs[index]);
        lhs_norm += static_cast<double>(lhs[index]) * static_cast<double>(lhs[index]);
        rhs_norm += static_cast<double>(rhs[index]) * static_cast<double>(rhs[index]);
    }
    if (lhs_norm <= 0.0 || rhs_norm <= 0.0) {
        return 1.0f;
    }
    return 1.0f - static_cast<float>(dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm)));
}

class MemoryStore final : public CorpusStore {
public:
    explicit MemoryStore(std::vector<ChunkRecord> chunks) : chunks_(std::move(chunks)) {}

    [[nodiscard]] bool is_open() const noexcept override { return true; }
    void close() noexcept override {}

    [[nodiscard]] godot_llama::Error set_embedding_state(const EmbeddingInfo &) override {
        return godot_llama::Error::make_ok();
    }

    [[nodiscard]] godot_llama::Error get_embedding_state(EmbeddingInfo &out_info, bool &out_present) const override {
        out_info.model_fingerprint = "mock-2";
        out_info.dimensions = 2;
        out_info.normalized = true;
        out_info.metric = VectorMetric::Cosine;
        out_present = true;
        return godot_llama::Error::make_ok();
    }

    [[nodiscard]] godot_llama::Error upsert_document(const SourceRecord &, const std::vector<ChunkRecord> &,
                                                     IngestStats &) override {
        return godot_llama::Error::make_ok();
    }

    [[nodiscard]] godot_llama::Error delete_source(std::string_view, IngestStats &) override {
        return godot_llama::Error::make_ok();
    }
    [[nodiscard]] godot_llama::Error clear(IngestStats &) override { return godot_llama::Error::make_ok(); }
    [[nodiscard]] godot_llama::Error list_sources(std::vector<SourceRecord> &) const override {
        return godot_llama::Error::make_ok();
    }
    [[nodiscard]] godot_llama::Error get_source(std::string_view, std::optional<SourceRecord> &) const override {
        return godot_llama::Error::make_ok();
    }

    [[nodiscard]] godot_llama::Error exact_vector_search(const std::vector<float> &query_vector,
                                                         const RetrievalOptions &options,
                                                         std::vector<VectorSearchHit> &out_hits) const override {
        out_hits.clear();
        for (const auto &chunk : chunks_) {
            if (!options.source_ids.empty() &&
                std::find(options.source_ids.begin(), options.source_ids.end(), chunk.source_id) ==
                        options.source_ids.end()) {
                continue;
            }
            if (!options.exclude_source_ids.empty() &&
                std::find(options.exclude_source_ids.begin(), options.exclude_source_ids.end(), chunk.source_id) !=
                        options.exclude_source_ids.end()) {
                continue;
            }
            if (!metadata_matches(chunk.metadata, options.metadata_filter)) {
                continue;
            }
            out_hits.push_back({chunk.chunk_id, cosine_distance(query_vector, chunk.embedding)});
        }

        std::sort(out_hits.begin(), out_hits.end(), [&](const VectorSearchHit &lhs, const VectorSearchHit &rhs) {
            if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
            }

            const auto lhs_chunk = std::find_if(chunks_.begin(), chunks_.end(), [&](const ChunkRecord &chunk) {
                return chunk.chunk_id == lhs.chunk_id;
            });
            const auto rhs_chunk = std::find_if(chunks_.begin(), chunks_.end(), [&](const ChunkRecord &chunk) {
                return chunk.chunk_id == rhs.chunk_id;
            });
            if (lhs_chunk != chunks_.end() && rhs_chunk != chunks_.end()) {
                if (lhs_chunk->source_id != rhs_chunk->source_id) {
                    return lhs_chunk->source_id < rhs_chunk->source_id;
                }
                return lhs_chunk->chunk_index < rhs_chunk->chunk_index;
            }
            return lhs.chunk_id < rhs.chunk_id;
        });

        if (static_cast<int32_t>(out_hits.size()) > options.candidate_k) {
            out_hits.resize(static_cast<size_t>(options.candidate_k));
        }
        return godot_llama::Error::make_ok();
    }

    [[nodiscard]] godot_llama::Error fetch_chunks_by_ids(const std::vector<std::string> &chunk_ids, bool include_embeddings,
                                                         std::vector<ChunkRecord> &out_chunks) const override {
        out_chunks.clear();
        for (const auto &chunk_id : chunk_ids) {
            const auto found = std::find_if(chunks_.begin(), chunks_.end(), [&](const ChunkRecord &chunk) {
                return chunk.chunk_id == chunk_id;
            });
            if (found == chunks_.end()) {
                return godot_llama::Error::make(godot_llama::ErrorCode::StorageCorrupt,
                                                "Requested chunk id is missing from the memory store", chunk_id);
            }

            ChunkRecord chunk = *found;
            if (!include_embeddings) {
                chunk.embedding.clear();
            }
            out_chunks.push_back(std::move(chunk));
        }
        return godot_llama::Error::make_ok();
    }

    [[nodiscard]] godot_llama::Error get_stats(CorpusStats &) const override { return godot_llama::Error::make_ok(); }

private:
    std::vector<ChunkRecord> chunks_;
};

ChunkRecord make_chunk(std::string chunk_id, std::string source_id, int64_t byte_start, int64_t byte_end,
                       std::vector<float> embedding, Metadata metadata = {}) {
    ChunkRecord chunk;
    chunk.chunk_id = std::move(chunk_id);
    chunk.source_id = std::move(source_id);
    chunk.title = chunk.source_id;
    chunk.display_text = chunk.chunk_id;
    chunk.byte_start = byte_start;
    chunk.byte_end = byte_end;
    chunk.embedding = std::move(embedding);
    chunk.embedding_info.dimensions = static_cast<int32_t>(chunk.embedding.size());
    chunk.embedding_info.metric = VectorMetric::Cosine;
    chunk.embedding_info.normalized = true;
    chunk.metadata = std::move(metadata);
    return chunk;
}

} // namespace

TEST_CASE("Dense retriever ranks, deduplicates, and diversifies results", "[rag][retriever]") {
    MockEmbedder embedder(2, true, VectorMetric::Cosine);
    embedder.set_vector("query alpha", {1.0f, 0.0f});

    std::vector<ChunkRecord> chunks = {
            make_chunk("chunk-a", "doc-a", 0, 10, {1.0f, 0.0f}, {{"topic", "alpha"}}),
            make_chunk("chunk-a-overlap", "doc-a", 4, 14, {0.95f, 0.05f}, {{"topic", "alpha"}}),
            make_chunk("chunk-b", "doc-b", 0, 10, {0.0f, 1.0f}, {{"topic", "beta"}}),
    };

    MemoryStore store(std::move(chunks));
    auto retriever = make_dense_retriever();
    auto reranker = make_noop_reranker();

    RetrievalOptions options;
    options.top_k = 2;
    options.candidate_k = 3;
    options.score_threshold = -1.0f;
    options.use_mmr = true;

    std::vector<RetrievalHit> hits;
    RetrievalStats stats;
    REQUIRE_FALSE(retriever->retrieve("query alpha", options, store, embedder, reranker.get(), hits, stats, {}));

    REQUIRE(hits.size() == 2);
    REQUIRE(hits[0].chunk_id == "chunk-a");
    REQUIRE(hits[1].chunk_id == "chunk-b");
    REQUIRE(stats.deduplicated_chunks >= 1);

    options.metadata_filter = {{"topic", "alpha"}};
    hits.clear();
    stats = {};
    REQUIRE_FALSE(retriever->retrieve("query alpha", options, store, embedder, reranker.get(), hits, stats, {}));
    REQUIRE(hits.size() == 1);
    REQUIRE(hits[0].source_id == "doc-a");
}
