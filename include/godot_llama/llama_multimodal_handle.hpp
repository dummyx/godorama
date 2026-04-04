#pragma once

#include "godot_llama/error.hpp"
#include "godot_llama/llama_params.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

struct llama_context;
struct mtmd_context;

namespace godot_llama {

class LlamaModelHandle;

class LlamaMultimodalHandle {
public:
    LlamaMultimodalHandle() noexcept = default;
    ~LlamaMultimodalHandle();

    LlamaMultimodalHandle(const LlamaMultimodalHandle &) = delete;
    LlamaMultimodalHandle &operator=(const LlamaMultimodalHandle &) = delete;
    LlamaMultimodalHandle(LlamaMultimodalHandle &&other) noexcept;
    LlamaMultimodalHandle &operator=(LlamaMultimodalHandle &&other) noexcept;

    [[nodiscard]] static Error create(const std::shared_ptr<LlamaModelHandle> &model, const MultimodalConfig &config,
                                      LlamaMultimodalHandle &out);

    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] bool supports_vision() const noexcept;
    [[nodiscard]] bool supports_audio() const noexcept;
    [[nodiscard]] int32_t audio_sample_rate_hz() const noexcept;
    [[nodiscard]] const std::string &media_marker() const noexcept;
    [[nodiscard]] Error evaluate_prompt(llama_context *lctx, std::string_view prompt,
                                        std::span<const MultimodalInput> media_inputs, bool add_special,
                                        int32_t n_batch, bool logits_last, int32_t &out_n_past) const;

private:
    mtmd_context *ctx_ = nullptr;
    std::string media_marker_ = kDefaultMediaMarker;
    bool supports_vision_ = false;
    bool supports_audio_ = false;
    int32_t audio_sample_rate_hz_ = -1;
};

} // namespace godot_llama
