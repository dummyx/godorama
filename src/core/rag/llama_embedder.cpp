#include "godot_llama/rag/factories.hpp"

#include "godot_llama/llama_context_handle.hpp"
#include "godot_llama/llama_model_handle.hpp"
#include "godot_llama/utf8.hpp"

#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace godot_llama::rag {
namespace {

void normalize_vector(std::vector<float> &vector) {
    double norm_sq = 0.0;
    for (const float value : vector) {
        norm_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    if (norm_sq <= 0.0) {
        return;
    }

    const float inv_norm = static_cast<float>(1.0 / std::sqrt(norm_sq));
    for (float &value : vector) {
        value *= inv_norm;
    }
}

class LlamaEmbedder final : public Embedder {
public:
    [[nodiscard]] Error open(const CorpusConfig &config) {
        ModelConfig embedding_config = config.embedding_model;
        embedding_config.embeddings_enabled = true;

        Error err = LlamaModelHandle::load(embedding_config, model_);
        if (err) {
            return err;
        }
        if (!model_) {
            return Error::make(ErrorCode::ModelLoadFailed, "Embedding model failed to load");
        }

        const ModelCapabilities &capabilities = model_->capabilities();
        if (!capabilities.supports_embeddings) {
            return Error::make(ErrorCode::EmbeddingsUnavailable,
                               "The configured embedding model does not expose embedding capabilities",
                               model_->descriptor());
        }

        err = LlamaContextHandle::create(model_, embedding_config, context_);
        if (err) {
            return err;
        }

        info_ = {};
        info_.model_fingerprint = model_->fingerprint();
        info_.normalize_output = config.normalize_embeddings;
        info_.metric = config.vector_metric;
        info_.pooling_type = context_.pooling_type();
        info_.supports_embeddings = capabilities.supports_embeddings;
        info_.supports_reranking = capabilities.supports_reranking;
        info_.dimensions = info_.pooling_type == LLAMA_POOLING_TYPE_RANK ? model_->n_cls_out()
                                                                          : std::max(model_->n_embd_out(), model_->n_embd());
        if (info_.dimensions <= 0) {
            return Error::make(ErrorCode::EmbeddingsUnavailable, "Embedding model reported an invalid vector dimension");
        }

        use_encoder_path_ = capabilities.has_encoder && !capabilities.has_decoder;
        return Error::make_ok();
    }

    [[nodiscard]] bool is_open() const noexcept override { return model_ && context_.is_valid(); }

    [[nodiscard]] const EmbedderInfo &info() const noexcept override { return info_; }

    [[nodiscard]] Error count_tokens(std::string_view text, int32_t &out_count) const override {
        if (!model_) {
            return Error::make(ErrorCode::NotOpen, "Embedding model is not loaded");
        }
        out_count = static_cast<int32_t>(model_->tokenize(text, true, false).size());
        return Error::make_ok();
    }

    [[nodiscard]] Error tokenize(std::string_view text, std::vector<int32_t> &out_tokens) const override {
        if (!model_) {
            return Error::make(ErrorCode::NotOpen, "Embedding model is not loaded");
        }
        out_tokens = model_->tokenize(text, true, false);
        return Error::make_ok();
    }

    [[nodiscard]] Error detokenize(const std::vector<int32_t> &tokens, std::string &out_text) const override {
        if (!model_) {
            return Error::make(ErrorCode::NotOpen, "Embedding model is not loaded");
        }
        out_text = model_->detokenize(tokens.data(), static_cast<int32_t>(tokens.size()));
        return Error::make_ok();
    }

    [[nodiscard]] Error embed(const std::vector<std::string> &texts, std::vector<std::vector<float>> &out_vectors,
                              const CancelCheck &is_cancelled) override {
        if (!is_open()) {
            return Error::make(ErrorCode::NotOpen, "Embedding model is not open");
        }

        out_vectors.clear();
        out_vectors.reserve(texts.size());

        for (const auto &text : texts) {
            if (is_cancelled && is_cancelled()) {
                return Error::make(ErrorCode::Cancelled, "Embedding cancelled");
            }
            if (!utf8::is_valid(text)) {
                return Error::make(ErrorCode::InvalidUtf8, "Embedding input is not valid UTF-8");
            }

            std::vector<int32_t> tokens = model_->tokenize(text, true, false);
            if (tokens.empty()) {
                return Error::make(ErrorCode::TokenizeFailed, "Embedding tokenization produced no tokens");
            }

            context_.clear_kv_cache();
            Error err = use_encoder_path_ ? context_.encode_tokens(tokens) : context_.decode_tokens(tokens, 0);
            if (err) {
                return err;
            }

            float *embedding_ptr = nullptr;
            if (info_.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                embedding_ptr = context_.get_embeddings_ith(-1);
                if (!embedding_ptr) {
                    embedding_ptr = context_.get_embeddings();
                }
            } else {
                embedding_ptr = context_.get_embeddings_seq(0);
            }

            if (!embedding_ptr) {
                return Error::make(ErrorCode::EmbeddingsUnavailable, "llama.cpp did not return an embedding vector");
            }

            std::vector<float> embedding(static_cast<size_t>(info_.dimensions));
            memcpy(embedding.data(), embedding_ptr, embedding.size() * sizeof(float));
            if (info_.normalize_output) {
                normalize_vector(embedding);
            }
            out_vectors.push_back(std::move(embedding));
        }

        return Error::make_ok();
    }

private:
    std::shared_ptr<LlamaModelHandle> model_;
    LlamaContextHandle context_;
    EmbedderInfo info_;
    bool use_encoder_path_ = false;
};

} // namespace

Error make_llama_embedder(const CorpusConfig &config, std::unique_ptr<Embedder> &out_embedder) {
    out_embedder.reset();
    auto embedder = std::make_unique<LlamaEmbedder>();
    Error err = embedder->open(config);
    if (err) {
        return err;
    }
    out_embedder = std::move(embedder);
    return Error::make_ok();
}

} // namespace godot_llama::rag
