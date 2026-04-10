#pragma once

#include "godot_llama/rag/types.hpp"

#include <memory>

namespace godot_llama {
class LlamaModelHandle;
}

namespace godot_llama::rag {

class TokenCounter {
public:
    virtual ~TokenCounter() = default;

    [[nodiscard]] virtual Error count_tokens(std::string_view text, int32_t &out_count) const = 0;
    [[nodiscard]] virtual Error tokenize(std::string_view text, std::vector<int32_t> &out_tokens) const = 0;
    [[nodiscard]] virtual Error detokenize(const std::vector<int32_t> &tokens, std::string &out_text) const = 0;
};

struct EmbedderInfo {
    std::string model_fingerprint;
    int32_t dimensions = 0;
    bool normalize_output = false;
    VectorMetric metric = VectorMetric::Cosine;
    int32_t pooling_type = 0;
    bool supports_embeddings = false;
    bool supports_reranking = false;
};

class Embedder : public TokenCounter {
public:
    ~Embedder() override = default;

    [[nodiscard]] virtual bool is_open() const noexcept = 0;
    [[nodiscard]] virtual const EmbedderInfo &info() const noexcept = 0;
    [[nodiscard]] virtual Error embed(const std::vector<std::string> &texts, std::vector<std::vector<float>> &out_vectors,
                                      const CancelCheck &is_cancelled) = 0;
};

class Chunker {
public:
    virtual ~Chunker() = default;

    [[nodiscard]] virtual Error chunk(const NormalizedDocument &document, const TokenCounter &token_counter,
                                      const ChunkingConfig &config, std::vector<ChunkRecord> &out_chunks) const = 0;
};

struct VectorSearchHit {
    std::string chunk_id;
    float distance = 0.0f;
};

class CorpusStore {
public:
    virtual ~CorpusStore() = default;

    [[nodiscard]] virtual bool is_open() const noexcept = 0;
    virtual void close() noexcept = 0;

    [[nodiscard]] virtual Error set_embedding_state(const EmbeddingInfo &embedding_info) = 0;
    [[nodiscard]] virtual Error get_embedding_state(EmbeddingInfo &out_info, bool &out_present) const = 0;

    [[nodiscard]] virtual Error upsert_document(const SourceRecord &source, const std::vector<ChunkRecord> &chunks,
                                                IngestStats &out_stats) = 0;
    [[nodiscard]] virtual Error delete_source(std::string_view source_id, IngestStats &out_stats) = 0;
    [[nodiscard]] virtual Error clear(IngestStats &out_stats) = 0;
    [[nodiscard]] virtual Error list_sources(std::vector<SourceRecord> &out_sources) const = 0;
    [[nodiscard]] virtual Error get_source(std::string_view source_id, std::optional<SourceRecord> &out_source) const = 0;
    [[nodiscard]] virtual Error exact_vector_search(const std::vector<float> &query_vector,
                                                    const RetrievalOptions &options,
                                                    std::vector<VectorSearchHit> &out_hits) const = 0;
    [[nodiscard]] virtual Error fetch_chunks_by_ids(const std::vector<std::string> &chunk_ids, bool include_embeddings,
                                                    std::vector<ChunkRecord> &out_chunks) const = 0;
    [[nodiscard]] virtual Error get_stats(CorpusStats &out_stats) const = 0;
};

class Reranker {
public:
    virtual ~Reranker() = default;

    [[nodiscard]] virtual bool is_available() const noexcept = 0;
    [[nodiscard]] virtual Error rerank(std::string_view query, std::vector<RetrievalHit> &hits,
                                       const CancelCheck &is_cancelled) const = 0;
    [[nodiscard]] virtual const char *status_name() const noexcept = 0;
};

class Retriever {
public:
    virtual ~Retriever() = default;

    [[nodiscard]] virtual Error retrieve(std::string_view query, const RetrievalOptions &options, const CorpusStore &store,
                                         Embedder &embedder, const Reranker *reranker,
                                         std::vector<RetrievalHit> &out_hits, RetrievalStats &out_stats,
                                         const CancelCheck &is_cancelled) const = 0;
};

class ContextPacker {
public:
    virtual ~ContextPacker() = default;

    [[nodiscard]] virtual Error assemble(std::string_view question, const RetrievalOptions &options,
                                         const std::vector<RetrievalHit> &hits,
                                         const std::shared_ptr<godot_llama::LlamaModelHandle> &generation_model,
                                         std::string_view chat_template_override, PromptAssembly &out_assembly) const = 0;
};

} // namespace godot_llama::rag
