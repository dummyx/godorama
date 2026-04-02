#include "godot_llama/rag/mock_embedder.hpp"

#include <algorithm>
#include <cmath>

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

} // namespace

MockEmbedder::MockEmbedder(int32_t dimensions, bool normalize_output, VectorMetric metric) {
    info_.dimensions = dimensions;
    info_.normalize_output = normalize_output;
    info_.metric = metric;
    info_.pooling_type = 0;
    info_.supports_embeddings = true;
    info_.supports_reranking = false;
    info_.model_fingerprint = "mock-" + std::to_string(dimensions);
}

void MockEmbedder::set_vector(std::string text, std::vector<float> vector) {
    if (info_.normalize_output) {
        normalize_vector(vector);
    }
    vectors_[std::move(text)] = std::move(vector);
}

bool MockEmbedder::is_open() const noexcept {
    return true;
}

const EmbedderInfo &MockEmbedder::info() const noexcept {
    return info_;
}

Error MockEmbedder::count_tokens(std::string_view text, int32_t &out_count) const {
    out_count = 0;
    bool in_token = false;
    for (const char ch : text) {
        if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') {
            in_token = false;
            continue;
        }
        if (!in_token) {
            ++out_count;
        }
        in_token = true;
    }
    return Error::make_ok();
}

Error MockEmbedder::tokenize(std::string_view text, std::vector<int32_t> &out_tokens) const {
    out_tokens.clear();
    int32_t current = 1;
    bool in_token = false;
    for (const char ch : text) {
        if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') {
            in_token = false;
            continue;
        }
        if (!in_token) {
            out_tokens.push_back(current++);
        }
        in_token = true;
    }
    return Error::make_ok();
}

Error MockEmbedder::detokenize(const std::vector<int32_t> &tokens, std::string &out_text) const {
    out_text.clear();
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (index > 0) {
            out_text.push_back(' ');
        }
        out_text.append("tok");
        out_text.append(std::to_string(tokens[index]));
    }
    return Error::make_ok();
}

Error MockEmbedder::embed(const std::vector<std::string> &texts, std::vector<std::vector<float>> &out_vectors,
                          const CancelCheck &is_cancelled) {
    out_vectors.clear();
    out_vectors.reserve(texts.size());

    for (const auto &text : texts) {
        if (is_cancelled && is_cancelled()) {
            return Error::make(ErrorCode::Cancelled, "Embedding cancelled");
        }

        const auto found = vectors_.find(text);
        if (found != vectors_.end()) {
            out_vectors.push_back(found->second);
            continue;
        }

        std::vector<float> fallback(static_cast<size_t>(std::max(info_.dimensions, 1)), 0.0f);
        int32_t index = 0;
        for (const unsigned char ch : text) {
            fallback[static_cast<size_t>(index % static_cast<int32_t>(fallback.size()))] +=
                    static_cast<float>((static_cast<int32_t>(ch) % 13) + 1);
            ++index;
        }
        if (info_.normalize_output) {
            normalize_vector(fallback);
        }
        out_vectors.push_back(std::move(fallback));
    }

    return Error::make_ok();
}

} // namespace godot_llama::rag
