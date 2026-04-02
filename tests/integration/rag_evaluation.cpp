#include "godot_llama/rag/corpus.hpp"
#include "godot_llama/rag/factories.hpp"
#include "godot_llama/rag/mock_embedder.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>

using namespace godot_llama::rag;

namespace {

std::filesystem::path evaluation_db_path() {
    const auto base = std::filesystem::temp_directory_path() / "godot_llama_rag_eval";
    std::filesystem::create_directories(base);
    const auto path = base / "fixture.sqlite3";
    std::filesystem::remove(path);
    return path;
}

std::unique_ptr<Embedder> make_eval_embedder() {
    auto embedder = std::make_unique<MockEmbedder>(2, true, VectorMetric::Cosine);
    embedder->set_vector("Godot uses scenes and nodes for game structure.\n", {1.0f, 0.0f});
    embedder->set_vector("SQLite stores structured local data in a single file.\n", {0.0f, 1.0f});
    embedder->set_vector("llama.cpp provides local inference for GGUF models.\n", {0.8f, 0.2f});
    embedder->set_vector("How does Godot organize a game?", {1.0f, 0.0f});
    embedder->set_vector("What stores local structured data?", {0.0f, 1.0f});
    embedder->set_vector("Which library runs local GGUF inference?", {0.8f, 0.2f});
    return embedder;
}

CorpusConfig make_eval_config(const std::filesystem::path &path) {
    CorpusConfig config;
    config.storage_path = path;
    config.chunking.chunk_size_tokens = 128;
    config.chunking.chunk_overlap_tokens = 0;
    config.normalize_embeddings = true;
    config.vector_metric = VectorMetric::Cosine;
    return config;
}

struct QueryCase {
    std::string query;
    std::string expected_source;
};

} // namespace

int main() {
    const auto path = evaluation_db_path();
    const CorpusConfig config = make_eval_config(path);

    std::unique_ptr<CorpusStore> store;
    if (auto err = make_sqlite_corpus_store(config, store)) {
        std::cerr << "failed to create store: " << err.message << "\n";
        return 1;
    }

    auto engine = std::make_unique<CorpusEngine>();
    if (auto err = engine->open(config, std::move(store), make_deterministic_chunker(), make_eval_embedder(),
                                make_dense_retriever(), make_noop_reranker())) {
        std::cerr << "failed to open engine: " << err.message << "\n";
        return 1;
    }

    const auto ingest_start = std::chrono::steady_clock::now();
    IngestStats ingest_stats;
    if (auto err = engine->upsert_text("doc-godot", "Godot uses scenes and nodes for game structure.\n", {}, ingest_stats,
                                       {}, {})) {
        std::cerr << err.message << "\n";
        return 1;
    }
    if (auto err = engine->upsert_text("doc-sqlite", "SQLite stores structured local data in a single file.\n", {},
                                       ingest_stats, {}, {})) {
        std::cerr << err.message << "\n";
        return 1;
    }
    if (auto err = engine->upsert_text("doc-llama", "llama.cpp provides local inference for GGUF models.\n", {},
                                       ingest_stats, {}, {})) {
        std::cerr << err.message << "\n";
        return 1;
    }
    const auto ingest_elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ingest_start).count();

    const std::vector<QueryCase> queries = {
            {"How does Godot organize a game?", "doc-godot"},
            {"What stores local structured data?", "doc-sqlite"},
            {"Which library runs local GGUF inference?", "doc-llama"},
    };

    int correct_at_2 = 0;
    double reciprocal_rank_sum = 0.0;
    double retrieval_ms = 0.0;

    for (const auto &query_case : queries) {
        const auto retrieve_start = std::chrono::steady_clock::now();
        std::vector<RetrievalHit> hits;
        RetrievalStats retrieval_stats;
        if (auto err = engine->retrieve(query_case.query, {}, hits, retrieval_stats, {})) {
            std::cerr << err.message << "\n";
            return 1;
        }
        retrieval_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - retrieve_start).count();

        bool matched = false;
        for (size_t rank = 0; rank < hits.size() && rank < 2; ++rank) {
            if (hits[rank].source_id == query_case.expected_source) {
                ++correct_at_2;
                reciprocal_rank_sum += 1.0 / static_cast<double>(rank + 1);
                matched = true;
                break;
            }
        }
        if (!matched) {
            reciprocal_rank_sum += 0.0;
        }
    }

    const double recall_at_2 = static_cast<double>(correct_at_2) / static_cast<double>(queries.size());
    const double mrr = reciprocal_rank_sum / static_cast<double>(queries.size());
    const double avg_retrieval_ms = retrieval_ms / static_cast<double>(queries.size());

    std::cout << "recall_at_2=" << recall_at_2 << "\n";
    std::cout << "mrr=" << mrr << "\n";
    std::cout << "ingest_ms=" << ingest_elapsed << "\n";
    std::cout << "avg_retrieve_ms=" << avg_retrieval_ms << "\n";
    std::cout << "answer_ms=0\n";

    if (recall_at_2 < 1.0 || mrr < 0.9) {
        return 1;
    }
    return 0;
}
