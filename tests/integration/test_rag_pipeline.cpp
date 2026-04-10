#include <catch2/catch_test_macros.hpp>

#include "godot_llama/rag/corpus.hpp"
#include "godot_llama/rag/factories.hpp"
#include "godot_llama/rag/mock_embedder.hpp"

#include <filesystem>

using namespace godot_llama::rag;

namespace {

std::filesystem::path unique_db_path(const char *name) {
    const auto base = std::filesystem::temp_directory_path() / "godot_llama_rag_tests";
    std::filesystem::create_directories(base);
    const auto path = base / name;
    std::filesystem::remove(path);
    return path;
}

std::unique_ptr<Embedder> make_fixture_embedder() {
    auto embedder = std::make_unique<MockEmbedder>(2, true, VectorMetric::Cosine);
    embedder->set_vector("Godot uses scenes and nodes for game structure.\n", {1.0f, 0.0f});
    embedder->set_vector("libSQL stores structured local data in an embedded database file.\n", {0.0f, 1.0f});
    embedder->set_vector("llama.cpp provides local inference for GGUF models.\n", {0.8f, 0.2f});
    embedder->set_vector("Where does Godot keep game structure?", {1.0f, 0.0f});
    return embedder;
}

CorpusConfig make_fixture_config(const std::filesystem::path &path) {
    CorpusConfig config;
    config.storage_path = path;
    config.chunking.chunk_size_tokens = 128;
    config.chunking.chunk_overlap_tokens = 0;
    config.normalize_embeddings = true;
    config.vector_metric = VectorMetric::Cosine;
    config.max_batch_texts = 8;
    return config;
}

} // namespace

TEST_CASE("libSQL-backed corpus persists across reopen and supports exact SQL retrieval", "[rag][integration]") {
    const auto path = unique_db_path("pipeline.db");
    const CorpusConfig config = make_fixture_config(path);

    {
        std::unique_ptr<CorpusStore> store;
        REQUIRE_FALSE(make_libsql_corpus_store(config, store));

        auto engine = std::make_unique<CorpusEngine>();
        REQUIRE_FALSE(engine->open(config, std::move(store), make_deterministic_chunker(), make_fixture_embedder(),
                                   make_dense_retriever(), make_noop_reranker()));

        IngestStats stats_a;
        REQUIRE_FALSE(engine->upsert_text("doc-godot", "Godot uses scenes and nodes for game structure.\n", {},
                                          stats_a, {}, {}));
        REQUIRE(stats_a.chunks_written == 1);

        IngestStats stats_b;
        REQUIRE_FALSE(engine->upsert_text("doc-libsql", "libSQL stores structured local data in an embedded database file.\n", {},
                                          stats_b, {}, {}));
        REQUIRE(stats_b.chunks_written == 1);

        std::vector<RetrievalHit> hits;
        RetrievalStats retrieval_stats;
        REQUIRE_FALSE(engine->retrieve("Where does Godot keep game structure?", {}, hits, retrieval_stats, {}));
        REQUIRE(hits.size() == 2);
        REQUIRE(hits.front().source_id == "doc-godot");
        REQUIRE(retrieval_stats.search_mode == "exact_sql");

        engine->close();
    }

    {
        std::unique_ptr<CorpusStore> store;
        REQUIRE_FALSE(make_libsql_corpus_store(config, store));

        auto engine = std::make_unique<CorpusEngine>();
        REQUIRE_FALSE(engine->open(config, std::move(store), make_deterministic_chunker(), make_fixture_embedder(),
                                   make_dense_retriever(), make_noop_reranker()));

        std::vector<RetrievalHit> hits;
        RetrievalStats retrieval_stats;
        REQUIRE_FALSE(engine->retrieve("Where does Godot keep game structure?", {}, hits, retrieval_stats, {}));
        REQUIRE_FALSE(hits.empty());
        REQUIRE(hits.front().source_id == "doc-godot");
        REQUIRE(retrieval_stats.search_mode == "exact_sql");

        IngestStats delete_stats;
        REQUIRE_FALSE(engine->delete_source("doc-libsql", delete_stats));
        REQUIRE(delete_stats.chunks_deleted == 1);

        engine->close();
    }
}
