#pragma once

#include "godot_llama/chat_template_engine.hpp"
#include "godot_llama/error.hpp"
#include "godot_llama/llama_lora_adapter_handle.hpp"
#include "godot_llama/llama_params.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct llama_context;
struct llama_model;
struct llama_vocab;

namespace godot_llama {

struct ModelCapabilities {
    bool supports_lora_adapters = true;
    bool has_encoder = false;
    bool has_decoder = false;
    bool supports_embeddings = false;
    bool supports_reranking = false;
    bool is_recurrent = false;
    bool is_hybrid = false;
    bool is_diffusion = false;
    int32_t n_ctx_train = 0;
    int32_t n_embd = 0;
    int32_t n_embd_out = 0;
    int32_t n_cls_out = 0;
    int32_t default_pooling_type = 0;
};

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
    [[nodiscard]] int32_t n_embd_inp() const noexcept;
    [[nodiscard]] int32_t n_embd_out() const noexcept;
    [[nodiscard]] int32_t n_cls_out() const noexcept;
    [[nodiscard]] int32_t n_vocab() const noexcept;

    [[nodiscard]] const ModelCapabilities &capabilities() const noexcept;
    [[nodiscard]] const std::string &descriptor() const noexcept;
    [[nodiscard]] const std::string &default_chat_template() const noexcept;
    [[nodiscard]] const std::string &fingerprint() const noexcept;
    [[nodiscard]] uint64_t model_size_bytes() const noexcept;
    [[nodiscard]] uint64_t parameter_count() const noexcept;
    [[nodiscard]] size_t lora_adapter_count() const noexcept;
    [[nodiscard]] std::optional<std::string> metadata_value(std::string_view key) const;
    [[nodiscard]] std::vector<std::pair<std::string, std::string>> metadata_entries() const;
    [[nodiscard]] Error apply_chat_template(const std::vector<std::pair<std::string, std::string>> &messages,
                                            bool add_assistant_turn, std::string_view template_override,
                                            bool disable_thinking,
                                            std::string &out_prompt) const;
    [[nodiscard]] Error apply_lora_adapters(llama_context *ctx) const;

    [[nodiscard]] std::vector<int32_t> tokenize(std::string_view text, bool add_bos, bool special) const;
    [[nodiscard]] std::string detokenize(const int32_t *tokens, int32_t n_tokens) const;
    [[nodiscard]] std::string token_to_piece(int32_t token) const;

private:
    [[nodiscard]] Error initialize_chat_template_engine(std::string_view template_override);
    [[nodiscard]] Error load_lora_adapters(const std::vector<LoraAdapterConfig> &configs);
    void refresh_metadata_cache();

    llama_model *model_ = nullptr;
    ModelCapabilities capabilities_;
    std::string descriptor_;
    std::string default_chat_template_;
    std::string fingerprint_;
    std::string configured_chat_template_override_;
    ChatTemplateEngine chat_template_engine_;
    std::vector<LlamaLoraAdapterHandle> lora_adapters_;
};

} // namespace godot_llama
