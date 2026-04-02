#include "godot_llama/rag/factories.hpp"

#include "godot_llama/rag/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace godot_llama::rag {
namespace {

float dot_product(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    const size_t size = std::min(lhs.size(), rhs.size());
    double value = 0.0;
    for (size_t index = 0; index < size; ++index) {
        value += static_cast<double>(lhs[index]) * static_cast<double>(rhs[index]);
    }
    return static_cast<float>(value);
}

float l2_norm(const std::vector<float> &value) {
    double sum = 0.0;
    for (const float item : value) {
        sum += static_cast<double>(item) * static_cast<double>(item);
    }
    return static_cast<float>(std::sqrt(sum));
}

float cosine_similarity(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    const float lhs_norm = l2_norm(lhs);
    const float rhs_norm = l2_norm(rhs);
    if (lhs_norm <= 0.0f || rhs_norm <= 0.0f) {
        return 0.0f;
    }
    return dot_product(lhs, rhs) / (lhs_norm * rhs_norm);
}

float overlap_ratio(const RetrievalHit &lhs, const RetrievalHit &rhs) {
    if (lhs.source_id != rhs.source_id) {
        return 0.0f;
    }

    const int64_t start = std::max(lhs.byte_start, rhs.byte_start);
    const int64_t end = std::min(lhs.byte_end, rhs.byte_end);
    if (end <= start) {
        return 0.0f;
    }

    const float intersection = static_cast<float>(end - start);
    const float lhs_length = static_cast<float>(std::max<int64_t>(1, lhs.byte_end - lhs.byte_start));
    const float rhs_length = static_cast<float>(std::max<int64_t>(1, rhs.byte_end - rhs.byte_start));
    return intersection / std::min(lhs_length, rhs_length);
}

struct ScoredCandidate {
    RetrievalHit hit;
    std::vector<float> embedding;
};

class DenseRetriever final : public Retriever {
public:
    [[nodiscard]] Error retrieve(std::string_view query, const RetrievalOptions &options, const CorpusStore &store,
                                 Embedder &embedder, const Reranker *reranker, std::vector<RetrievalHit> &out_hits,
                                 RetrievalStats &out_stats, const CancelCheck &is_cancelled) const override {
        out_hits.clear();
        out_stats = {};
        if (query.empty()) {
            return Error::make(ErrorCode::InvalidParameter, "Query is empty");
        }

        std::vector<ChunkRecord> candidates;
        Error err = store.fetch_candidate_chunks(options, candidates);
        if (err) {
            return err;
        }
        out_stats.scanned_chunks = static_cast<int32_t>(candidates.size());

        int32_t query_token_count = 0;
        err = embedder.count_tokens(query, query_token_count);
        if (err) {
            return err;
        }
        out_stats.query_token_count = query_token_count;

        std::vector<std::vector<float>> query_embeddings;
        err = embedder.embed({std::string(query)}, query_embeddings, is_cancelled);
        if (err) {
            return err;
        }
        if (query_embeddings.empty()) {
            return Error::make(ErrorCode::EmbeddingsUnavailable, "Failed to embed query");
        }

        const auto &query_embedding = query_embeddings.front();
        std::vector<ScoredCandidate> scored;
        scored.reserve(candidates.size());

        float min_score = 0.0f;
        float max_score = 0.0f;
        bool first_score = true;

        for (const auto &chunk : candidates) {
            if (is_cancelled && is_cancelled()) {
                return Error::make(ErrorCode::Cancelled, "Retrieval cancelled");
            }
            if (chunk.embedding.empty()) {
                ++out_stats.filtered_chunks;
                continue;
            }

            const bool cosine = chunk.embedding_info.metric == VectorMetric::Cosine || chunk.embedding_info.normalized;
            const float raw_score =
                    cosine ? cosine_similarity(query_embedding, chunk.embedding) : dot_product(query_embedding, chunk.embedding);

            RetrievalHit hit;
            hit.chunk_id = chunk.chunk_id;
            hit.source_id = chunk.source_id;
            hit.title = chunk.title;
            hit.source_path = chunk.source_path;
            hit.excerpt = chunk.display_text;
            hit.metadata = chunk.metadata;
            hit.raw_score = raw_score;
            hit.byte_start = chunk.byte_start;
            hit.byte_end = chunk.byte_end;
            hit.char_start = chunk.char_start;
            hit.char_end = chunk.char_end;
            hit.token_count = chunk.token_count;

            if (first_score) {
                min_score = raw_score;
                max_score = raw_score;
                first_score = false;
            } else {
                min_score = std::min(min_score, raw_score);
                max_score = std::max(max_score, raw_score);
            }

            scored.push_back({std::move(hit), chunk.embedding});
        }

        const float range = max_score - min_score;
        for (auto &candidate : scored) {
            if (candidate.hit.raw_score < options.score_threshold) {
                continue;
            }

            if (candidate.hit.raw_score >= -1.0f && candidate.hit.raw_score <= 1.0f) {
                candidate.hit.relevance_score = std::clamp((candidate.hit.raw_score + 1.0f) * 0.5f, 0.0f, 1.0f);
            } else if (range > 0.0f) {
                candidate.hit.relevance_score = (candidate.hit.raw_score - min_score) / range;
            } else {
                candidate.hit.relevance_score = 1.0f;
            }
        }

        scored.erase(std::remove_if(scored.begin(), scored.end(), [&](const ScoredCandidate &candidate) {
                         return candidate.hit.raw_score < options.score_threshold;
                     }),
                     scored.end());

        std::sort(scored.begin(), scored.end(), [](const ScoredCandidate &lhs, const ScoredCandidate &rhs) {
            return lhs.hit.relevance_score > rhs.hit.relevance_score;
        });

        if (static_cast<int32_t>(scored.size()) > options.candidate_k) {
            scored.resize(static_cast<size_t>(options.candidate_k));
        }
        out_stats.candidate_chunks = static_cast<int32_t>(scored.size());

        if (options.use_reranker && reranker && reranker->is_available()) {
            std::vector<RetrievalHit> rerank_hits;
            rerank_hits.reserve(scored.size());
            for (const auto &candidate : scored) {
                rerank_hits.push_back(candidate.hit);
            }

            err = reranker->rerank(query, rerank_hits, is_cancelled);
            if (err) {
                return err;
            }

            for (size_t index = 0; index < rerank_hits.size(); ++index) {
                scored[index].hit = rerank_hits[index];
            }

            std::sort(scored.begin(), scored.end(), [](const ScoredCandidate &lhs, const ScoredCandidate &rhs) {
                const float lhs_score = lhs.hit.rerank_score.value_or(lhs.hit.relevance_score);
                const float rhs_score = rhs.hit.rerank_score.value_or(rhs.hit.relevance_score);
                return lhs_score > rhs_score;
            });
            out_stats.reranker_used = true;
            out_stats.reranker_status = reranker->status_name();
        } else if (options.use_reranker && reranker) {
            out_stats.reranker_status = reranker->status_name();
        } else {
            out_stats.reranker_status = "disabled";
        }

        std::vector<ScoredCandidate> deduplicated;
        deduplicated.reserve(scored.size());
        for (const auto &candidate : scored) {
            bool duplicate = false;
            for (const auto &existing : deduplicated) {
                if (candidate.hit.source_id == existing.hit.source_id &&
                    overlap_ratio(candidate.hit, existing.hit) >= 0.6f) {
                    duplicate = true;
                    ++out_stats.deduplicated_chunks;
                    break;
                }
            }
            if (!duplicate) {
                deduplicated.push_back(candidate);
            }
        }

        std::vector<ScoredCandidate> selected;
        selected.reserve(static_cast<size_t>(options.top_k));

        if (options.use_mmr) {
            std::vector<ScoredCandidate> remaining = deduplicated;
            while (!remaining.empty() && static_cast<int32_t>(selected.size()) < options.top_k) {
                size_t best_index = 0;
                float best_score = -std::numeric_limits<float>::infinity();

                for (size_t index = 0; index < remaining.size(); ++index) {
                    float redundancy = 0.0f;
                    for (const auto &chosen : selected) {
                        redundancy = std::max(redundancy, cosine_similarity(remaining[index].embedding, chosen.embedding));
                    }

                    const float utility = (options.mmr_lambda * remaining[index].hit.relevance_score) -
                                          ((1.0f - options.mmr_lambda) * redundancy);
                    if (utility > best_score) {
                        best_score = utility;
                        best_index = index;
                    }
                }

                selected.push_back(remaining[best_index]);
                remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(best_index));
            }
        } else {
            selected = deduplicated;
            if (static_cast<int32_t>(selected.size()) > options.top_k) {
                selected.resize(static_cast<size_t>(options.top_k));
            }
        }

        out_hits.reserve(selected.size());
        for (const auto &candidate : selected) {
            out_hits.push_back(candidate.hit);
        }
        out_stats.returned_chunks = static_cast<int32_t>(out_hits.size());
        return Error::make_ok();
    }
};

} // namespace

std::unique_ptr<Retriever> make_dense_retriever() {
    return std::make_unique<DenseRetriever>();
}

} // namespace godot_llama::rag
