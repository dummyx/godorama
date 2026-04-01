#pragma once

#include "godot_llama/llama_params.hpp"

#include <cstdint>

struct llama_sampler;
struct llama_vocab;
struct llama_context;

namespace godot_llama {

class LlamaSamplerHandle {
public:
    LlamaSamplerHandle() noexcept = default;
    ~LlamaSamplerHandle();

    LlamaSamplerHandle(const LlamaSamplerHandle &) = delete;
    LlamaSamplerHandle &operator=(const LlamaSamplerHandle &) = delete;
    LlamaSamplerHandle(LlamaSamplerHandle &&other) noexcept;
    LlamaSamplerHandle &operator=(LlamaSamplerHandle &&other) noexcept;

    void init(const GenerateOptions &opts, const llama_vocab *vocab);
    void reset() noexcept;

    [[nodiscard]] int32_t sample(::llama_context *ctx, int32_t idx);
    void accept(int32_t token) noexcept;

    [[nodiscard]] bool is_valid() const noexcept;

private:
    llama_sampler *chain_ = nullptr;
};

} // namespace godot_llama
