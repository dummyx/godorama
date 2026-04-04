#pragma once

#include "godot_llama/error.hpp"
#include "godot_llama/llama_params.hpp"

#include <string>

struct llama_adapter_lora;
struct llama_model;

namespace godot_llama {

class LlamaLoraAdapterHandle {
public:
    LlamaLoraAdapterHandle() noexcept = default;
    ~LlamaLoraAdapterHandle();

    LlamaLoraAdapterHandle(const LlamaLoraAdapterHandle &) = delete;
    LlamaLoraAdapterHandle &operator=(const LlamaLoraAdapterHandle &) = delete;
    LlamaLoraAdapterHandle(LlamaLoraAdapterHandle &&other) noexcept;
    LlamaLoraAdapterHandle &operator=(LlamaLoraAdapterHandle &&other) noexcept;

    [[nodiscard]] static Error load(llama_model *model, const LoraAdapterConfig &config, LlamaLoraAdapterHandle &out);

    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] llama_adapter_lora *raw() const noexcept;
    [[nodiscard]] float scale() const noexcept;
    [[nodiscard]] const std::string &path() const noexcept;

private:
    llama_adapter_lora *adapter_ = nullptr;
    std::string path_;
    float scale_ = 1.0f;
};

} // namespace godot_llama
