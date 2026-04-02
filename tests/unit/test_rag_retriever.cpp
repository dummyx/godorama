#include <catch2/catch_test_macros.hpp>

#include "godot_llama/rag/factories.hpp"
#include "godot_llama/rag/mock_embedder.hpp"

using namespace godot_llama::rag;

namespace {

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

    [[nodiscard]] godot_llama::Error fetch_candidate_chunks(const RetrievalOptions &options,
                                                            std::vector<ChunkRecord> &out_chunks) const override {
        out_chunks.clear();
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
            out_chunks.push_back(chunk);
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
