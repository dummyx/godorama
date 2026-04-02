#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace godot_llama {

struct ModelConfig {
    std::string model_path;
    int32_t n_ctx = 2048;
    int32_t n_threads = -1; // -1 = auto
    int32_t n_batch = 512;
    int32_t n_gpu_layers = 0;
    uint32_t seed = 0xFFFFFFFF;
    bool use_mmap = true;
    bool use_mlock = false;
    bool embeddings_enabled = false;
    bool disable_thinking = false;
    std::string chat_template_override;
};

struct GenerateOptions {
    int32_t max_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int32_t top_k = 40;
    float min_p = 0.05f;
    float repeat_penalty = 1.1f;
    int32_t repeat_last_n = 64;
    std::vector<std::string> stop;
    std::optional<uint32_t> seed_override;
};

} // namespace godot_llama
