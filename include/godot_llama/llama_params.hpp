#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace godot_llama {

inline constexpr char kDefaultMediaMarker[] = "<__media__>";

struct LoraAdapterConfig {
    std::string path;
    float scale = 1.0f;
};

struct MultimodalConfig {
    std::string mmproj_path;
    std::string media_marker = kDefaultMediaMarker;
    bool use_gpu = false;
    bool print_timings = false;
    int32_t n_threads = -1; // -1 = auto
    int32_t image_min_tokens = 0;
    int32_t image_max_tokens = 0;
};

enum class MultimodalInputType : uint8_t {
    Image = 0,
    Audio = 1,
};

struct MultimodalInput {
    MultimodalInputType type = MultimodalInputType::Image;
    std::string path;
    std::string id;
    std::vector<uint8_t> data; // in-memory buffer; when non-empty, takes precedence over path
};

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
    int32_t flash_attn_type = -1; // -1 = auto, 0 = disabled, 1 = enabled
    int32_t type_k = -1;          // KV cache key type: -1 = default (F16), see ggml_type
    int32_t type_v = -1;          // KV cache value type: -1 = default (F16), see ggml_type
    std::string chat_template_override;
    std::vector<LoraAdapterConfig> lora_adapters;
    std::optional<MultimodalConfig> multimodal;
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
