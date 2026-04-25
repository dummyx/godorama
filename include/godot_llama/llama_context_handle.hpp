#pragma once

#include "godot_llama/error.hpp"
#include "godot_llama/llama_params.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

struct llama_context;
struct llama_batch;

namespace godot_llama {

class LlamaModelHandle;

class LlamaContextHandle {
public:
    LlamaContextHandle() noexcept = default;
    ~LlamaContextHandle();

    LlamaContextHandle(const LlamaContextHandle &) = delete;
    LlamaContextHandle &operator=(const LlamaContextHandle &) = delete;
    LlamaContextHandle(LlamaContextHandle &&other) noexcept;
    LlamaContextHandle &operator=(LlamaContextHandle &&other) noexcept;

    [[nodiscard]] static Error create(const std::shared_ptr<LlamaModelHandle> &model, const ModelConfig &config,
                                      LlamaContextHandle &out);

    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] llama_context *raw() const noexcept;
    [[nodiscard]] const std::shared_ptr<LlamaModelHandle> &model() const noexcept;

    [[nodiscard]] Error encode_tokens(std::span<const int32_t> tokens);
    [[nodiscard]] Error decode_tokens(std::span<const int32_t> tokens, int32_t pos_offset);
    [[nodiscard]] Error decode_embeddings(std::span<const float> embeddings, int32_t token_count, int32_t hidden_size,
                                          std::span<const int32_t> positions, int32_t position_components,
                                          int32_t seq_id = 0);
    [[nodiscard]] float *get_logits(int32_t idx) const noexcept;
    [[nodiscard]] float *get_embeddings() const noexcept;
    [[nodiscard]] float *get_embeddings_ith(int32_t index) const noexcept;
    [[nodiscard]] float *get_embeddings_seq(int32_t seq_id) const noexcept;
    [[nodiscard]] int32_t pooling_type() const noexcept;

    void set_abort_flag(const std::atomic<bool> *abort_flag) noexcept;
    void clear_kv_cache() noexcept;
    [[nodiscard]] int32_t n_ctx() const noexcept;

private:
    struct LlamaBatchDeleter {
        void operator()(llama_batch *batch) const noexcept;
    };

    void free_embedding_batch() noexcept;
    [[nodiscard]] Error ensure_embedding_batch(int32_t token_count, int32_t hidden_size);

    llama_context *ctx_ = nullptr;
    std::unique_ptr<llama_batch, LlamaBatchDeleter> embedding_batch_;
    int32_t embedding_batch_token_capacity_ = 0;
    int32_t embedding_batch_hidden_size_ = 0;
    std::shared_ptr<LlamaModelHandle> model_;
    const std::atomic<bool> *abort_flag_ = nullptr;
    bool embeddings_enabled_ = false;
};

} // namespace godot_llama
