#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>

namespace godot {

class LlamaModelConfig : public Resource {
    GDCLASS(LlamaModelConfig, Resource)

public:
    LlamaModelConfig() = default;
    ~LlamaModelConfig() override = default;

    void set_model_path(const String &p_path);
    String get_model_path() const;

    void set_n_ctx(int32_t p_n_ctx);
    int32_t get_n_ctx() const;

    void set_n_threads(int32_t p_n_threads);
    int32_t get_n_threads() const;

    void set_n_batch(int32_t p_n_batch);
    int32_t get_n_batch() const;

    void set_n_gpu_layers(int32_t p_n_gpu_layers);
    int32_t get_n_gpu_layers() const;

    void set_seed(int32_t p_seed);
    int32_t get_seed() const;

    void set_use_mmap(bool p_use_mmap);
    bool get_use_mmap() const;

    void set_use_mlock(bool p_use_mlock);
    bool get_use_mlock() const;

    void set_embeddings_enabled(bool p_enabled);
    bool get_embeddings_enabled() const;

    void set_disable_thinking(bool p_disable_thinking);
    bool get_disable_thinking() const;

    void set_chat_template_override(const String &p_template);
    String get_chat_template_override() const;

protected:
    static void _bind_methods();

private:
    String model_path_;
    int32_t n_ctx_ = 2048;
    int32_t n_threads_ = -1;
    int32_t n_batch_ = 512;
    int32_t n_gpu_layers_ = 0;
    int32_t seed_ = static_cast<int32_t>(0xFFFFFFFF);
    bool use_mmap_ = true;
    bool use_mlock_ = false;
    bool embeddings_enabled_ = false;
    bool disable_thinking_ = false;
    String chat_template_override_;
};

} // namespace godot
