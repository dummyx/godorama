#pragma once

#include "godot_llama/rag/interfaces.hpp"

#include <unordered_map>

namespace godot_llama::rag {

class MockEmbedder final : public Embedder {
public:
    MockEmbedder(int32_t dimensions, bool normalize_output, VectorMetric metric);

    void set_vector(std::string text, std::vector<float> vector);

    [[nodiscard]] bool is_open() const noexcept override;
    [[nodiscard]] const EmbedderInfo &info() const noexcept override;

    [[nodiscard]] Error count_tokens(std::string_view text, int32_t &out_count) const override;
    [[nodiscard]] Error tokenize(std::string_view text, std::vector<int32_t> &out_tokens) const override;
    [[nodiscard]] Error detokenize(const std::vector<int32_t> &tokens, std::string &out_text) const override;
    [[nodiscard]] Error embed(const std::vector<std::string> &texts, std::vector<std::vector<float>> &out_vectors,
                              const CancelCheck &is_cancelled) override;

private:
    EmbedderInfo info_;
    std::unordered_map<std::string, std::vector<float>> vectors_;
};

} // namespace godot_llama::rag
