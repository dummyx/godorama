#include <catch2/catch_test_macros.hpp>

#include "godot_llama/rag/factories.hpp"
#include "godot_llama/rag/mock_embedder.hpp"

using namespace godot_llama::rag;

TEST_CASE("Deterministic chunker preserves stable chunk boundaries", "[rag][chunker]") {
    MockEmbedder token_counter(4, false, VectorMetric::Cosine);
    NormalizedDocument document;
    document.source_id = "doc-a";
    document.source_version = "v1";
    document.title = "Doc A";
    document.normalized_text =
            "# Title\n\nalpha beta gamma delta\n\nsecond paragraph epsilon zeta eta theta\n";

    auto chunker = make_deterministic_chunker();
    ChunkingConfig config;
    config.chunk_size_tokens = 4;
    config.chunk_overlap_tokens = 1;

    std::vector<ChunkRecord> first;
    std::vector<ChunkRecord> second;

    REQUIRE_FALSE(chunker->chunk(document, token_counter, config, first));
    REQUIRE_FALSE(chunker->chunk(document, token_counter, config, second));

    REQUIRE(first.size() >= 2);
    REQUIRE(first.size() == second.size());
    REQUIRE(first.front().byte_start == 0);
    REQUIRE(first.back().byte_end == static_cast<int64_t>(document.normalized_text.size()));

    for (size_t index = 0; index < first.size(); ++index) {
        REQUIRE(first[index].chunk_id == second[index].chunk_id);
        REQUIRE(first[index].token_count <= config.chunk_size_tokens);
        REQUIRE(first[index].byte_end > first[index].byte_start);
        REQUIRE(first[index].char_end > first[index].char_start);
    }
}
