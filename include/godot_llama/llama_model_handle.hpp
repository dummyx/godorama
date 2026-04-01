#pragma once

#include "godot_llama/error.hpp"
#include "godot_llama/llama_params.hpp"

#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;

namespace godot_llama {

class LlamaModelHandle {
public:
    LlamaModelHandle() noexcept = default;
    ~LlamaModelHandle();

    LlamaModelHandle(const LlamaModelHandle &) = delete;
    LlamaModelHandle &operator=(const LlamaModelHandle &) = delete;
    LlamaModelHandle(LlamaModelHandle &&other) noexcept;
    LlamaModelHandle &operator=(LlamaModelHandle &&other) noexcept;

    [[nodiscard]] static Error load(const ModelConfig &config, std::shared_ptr<LlamaModelHandle> &out);

    [[nodiscard]] bool is_loaded() const noexcept;
    [[nodiscard]] llama_model *raw() const noexcept;
    [[nodiscard]] const llama_vocab *vocab() const noexcept;

    [[nodiscard]] int32_t n_ctx_train() const noexcept;
    [[nodiscard]] int32_t n_embd() const noexcept;

    [[nodiscard]] std::vector<int32_t> tokenize(std::string_view text, bool add_bos, bool special) const;
    [[nodiscard]] std::string detokenize(const int32_t *tokens, int32_t n_tokens) const;
    [[nodiscard]] std::string token_to_piece(int32_t token) const;

private:
    llama_model *model_ = nullptr;
};

} // namespace godot_llama
