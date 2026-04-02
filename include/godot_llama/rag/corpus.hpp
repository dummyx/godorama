#pragma once

#include "godot_llama/rag/interfaces.hpp"

#include <mutex>
#include <memory>
#include <optional>

namespace godot_llama::rag {

class CorpusEngine {
public:
    CorpusEngine();
    ~CorpusEngine();

    CorpusEngine(const CorpusEngine &) = delete;
    CorpusEngine &operator=(const CorpusEngine &) = delete;

    [[nodiscard]] Error open(const CorpusConfig &config, std::unique_ptr<CorpusStore> store,
                             std::unique_ptr<Chunker> chunker, std::unique_ptr<Embedder> embedder,
                             std::unique_ptr<Retriever> retriever, std::unique_ptr<Reranker> reranker);
    void close() noexcept;

    [[nodiscard]] bool is_open() const noexcept;
    [[nodiscard]] const CorpusConfig &config() const noexcept;

    [[nodiscard]] Error upsert_text(std::string source_id, std::string text, Metadata metadata, IngestStats &out_stats,
                                    const ProgressCallback &on_progress, const CancelCheck &is_cancelled);
    [[nodiscard]] Error upsert_file(const std::filesystem::path &path, Metadata metadata, IngestStats &out_stats,
                                    const ProgressCallback &on_progress, const CancelCheck &is_cancelled);
    [[nodiscard]] Error delete_source(std::string_view source_id, IngestStats &out_stats);
    [[nodiscard]] Error clear(IngestStats &out_stats);
    [[nodiscard]] Error rebuild(IngestStats &out_stats, const ProgressCallback &on_progress,
                                const CancelCheck &is_cancelled);
    [[nodiscard]] Error retrieve(std::string_view query, const RetrievalOptions &options,
                                 std::vector<RetrievalHit> &out_hits, RetrievalStats &out_stats,
                                 const CancelCheck &is_cancelled);
    [[nodiscard]] Error get_stats(CorpusStats &out_stats) const;

private:
    [[nodiscard]] Error ensure_open() const;
    [[nodiscard]] Error ensure_embedding_fingerprint_consistency(bool allow_empty_corpus) const;
    [[nodiscard]] Error normalize_document(std::string source_id, std::string title, std::string source_path,
                                           std::string text, Metadata metadata, ParserMode parser_mode,
                                           NormalizedDocument &out_document) const;
    [[nodiscard]] Error ingest_document(const NormalizedDocument &document, IngestStats &out_stats,
                                        const ProgressCallback &on_progress, const CancelCheck &is_cancelled);

    CorpusConfig config_;
    bool open_ = false;
    std::unique_ptr<CorpusStore> store_;
    std::unique_ptr<Chunker> chunker_;
    std::unique_ptr<Embedder> embedder_;
    std::unique_ptr<Retriever> retriever_;
    std::unique_ptr<Reranker> reranker_;
    mutable std::mutex mutex_;
};

} // namespace godot_llama::rag
