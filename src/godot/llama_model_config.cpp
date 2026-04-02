#include "llama_model_config.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

void LlamaModelConfig::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_model_path", "path"), &LlamaModelConfig::set_model_path);
    ClassDB::bind_method(D_METHOD("get_model_path"), &LlamaModelConfig::get_model_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "model_path", PROPERTY_HINT_FILE, "*.gguf"), "set_model_path",
                 "get_model_path");

    ClassDB::bind_method(D_METHOD("set_n_ctx", "n_ctx"), &LlamaModelConfig::set_n_ctx);
    ClassDB::bind_method(D_METHOD("get_n_ctx"), &LlamaModelConfig::get_n_ctx);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "n_ctx"), "set_n_ctx", "get_n_ctx");

    ClassDB::bind_method(D_METHOD("set_n_threads", "n_threads"), &LlamaModelConfig::set_n_threads);
    ClassDB::bind_method(D_METHOD("get_n_threads"), &LlamaModelConfig::get_n_threads);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "n_threads"), "set_n_threads", "get_n_threads");

    ClassDB::bind_method(D_METHOD("set_n_batch", "n_batch"), &LlamaModelConfig::set_n_batch);
    ClassDB::bind_method(D_METHOD("get_n_batch"), &LlamaModelConfig::get_n_batch);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "n_batch"), "set_n_batch", "get_n_batch");

    ClassDB::bind_method(D_METHOD("set_n_gpu_layers", "n_gpu_layers"), &LlamaModelConfig::set_n_gpu_layers);
    ClassDB::bind_method(D_METHOD("get_n_gpu_layers"), &LlamaModelConfig::get_n_gpu_layers);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "n_gpu_layers"), "set_n_gpu_layers", "get_n_gpu_layers");

    ClassDB::bind_method(D_METHOD("set_seed", "seed"), &LlamaModelConfig::set_seed);
    ClassDB::bind_method(D_METHOD("get_seed"), &LlamaModelConfig::get_seed);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "seed"), "set_seed", "get_seed");

    ClassDB::bind_method(D_METHOD("set_use_mmap", "use_mmap"), &LlamaModelConfig::set_use_mmap);
    ClassDB::bind_method(D_METHOD("get_use_mmap"), &LlamaModelConfig::get_use_mmap);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_mmap"), "set_use_mmap", "get_use_mmap");

    ClassDB::bind_method(D_METHOD("set_use_mlock", "use_mlock"), &LlamaModelConfig::set_use_mlock);
    ClassDB::bind_method(D_METHOD("get_use_mlock"), &LlamaModelConfig::get_use_mlock);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_mlock"), "set_use_mlock", "get_use_mlock");

    ClassDB::bind_method(D_METHOD("set_embeddings_enabled", "enabled"), &LlamaModelConfig::set_embeddings_enabled);
    ClassDB::bind_method(D_METHOD("get_embeddings_enabled"), &LlamaModelConfig::get_embeddings_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "embeddings_enabled"), "set_embeddings_enabled", "get_embeddings_enabled");

    ClassDB::bind_method(D_METHOD("set_disable_thinking", "disable_thinking"), &LlamaModelConfig::set_disable_thinking);
    ClassDB::bind_method(D_METHOD("get_disable_thinking"), &LlamaModelConfig::get_disable_thinking);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disable_thinking"), "set_disable_thinking", "get_disable_thinking");

    ClassDB::bind_method(D_METHOD("set_chat_template_override", "tmpl"), &LlamaModelConfig::set_chat_template_override);
    ClassDB::bind_method(D_METHOD("get_chat_template_override"), &LlamaModelConfig::get_chat_template_override);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "chat_template_override"), "set_chat_template_override",
                 "get_chat_template_override");
}

void LlamaModelConfig::set_model_path(const String &p_path) {
    model_path_ = p_path;
}
String LlamaModelConfig::get_model_path() const {
    return model_path_;
}

void LlamaModelConfig::set_n_ctx(int32_t p_n_ctx) {
    n_ctx_ = p_n_ctx > 0 ? p_n_ctx : 2048;
}
int32_t LlamaModelConfig::get_n_ctx() const {
    return n_ctx_;
}

void LlamaModelConfig::set_n_threads(int32_t p_n_threads) {
    n_threads_ = p_n_threads;
}
int32_t LlamaModelConfig::get_n_threads() const {
    return n_threads_;
}

void LlamaModelConfig::set_n_batch(int32_t p_n_batch) {
    n_batch_ = p_n_batch > 0 ? p_n_batch : 512;
}
int32_t LlamaModelConfig::get_n_batch() const {
    return n_batch_;
}

void LlamaModelConfig::set_n_gpu_layers(int32_t p_n_gpu_layers) {
    n_gpu_layers_ = p_n_gpu_layers;
}
int32_t LlamaModelConfig::get_n_gpu_layers() const {
    return n_gpu_layers_;
}

void LlamaModelConfig::set_seed(int32_t p_seed) {
    seed_ = p_seed;
}
int32_t LlamaModelConfig::get_seed() const {
    return seed_;
}

void LlamaModelConfig::set_use_mmap(bool p_use_mmap) {
    use_mmap_ = p_use_mmap;
}
bool LlamaModelConfig::get_use_mmap() const {
    return use_mmap_;
}

void LlamaModelConfig::set_use_mlock(bool p_use_mlock) {
    use_mlock_ = p_use_mlock;
}
bool LlamaModelConfig::get_use_mlock() const {
    return use_mlock_;
}

void LlamaModelConfig::set_embeddings_enabled(bool p_enabled) {
    embeddings_enabled_ = p_enabled;
}
bool LlamaModelConfig::get_embeddings_enabled() const {
    return embeddings_enabled_;
}

void LlamaModelConfig::set_disable_thinking(bool p_disable_thinking) {
    disable_thinking_ = p_disable_thinking;
}
bool LlamaModelConfig::get_disable_thinking() const {
    return disable_thinking_;
}

void LlamaModelConfig::set_chat_template_override(const String &p_template) {
    chat_template_override_ = p_template;
}
String LlamaModelConfig::get_chat_template_override() const {
    return chat_template_override_;
}

} // namespace godot
