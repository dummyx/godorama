#include "llama_multimodal_config.hpp"

#include <godot_cpp/core/class_db.hpp>

namespace godot {

void LlamaMultimodalConfig::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_mmproj_path", "path"), &LlamaMultimodalConfig::set_mmproj_path);
    ClassDB::bind_method(D_METHOD("get_mmproj_path"), &LlamaMultimodalConfig::get_mmproj_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "mmproj_path", PROPERTY_HINT_FILE, "*.gguf"), "set_mmproj_path",
                 "get_mmproj_path");

    ClassDB::bind_method(D_METHOD("set_media_marker", "marker"), &LlamaMultimodalConfig::set_media_marker);
    ClassDB::bind_method(D_METHOD("get_media_marker"), &LlamaMultimodalConfig::get_media_marker);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "media_marker"), "set_media_marker", "get_media_marker");

    ClassDB::bind_method(D_METHOD("set_use_gpu", "use_gpu"), &LlamaMultimodalConfig::set_use_gpu);
    ClassDB::bind_method(D_METHOD("get_use_gpu"), &LlamaMultimodalConfig::get_use_gpu);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gpu"), "set_use_gpu", "get_use_gpu");

    ClassDB::bind_method(D_METHOD("set_print_timings", "print_timings"), &LlamaMultimodalConfig::set_print_timings);
    ClassDB::bind_method(D_METHOD("get_print_timings"), &LlamaMultimodalConfig::get_print_timings);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "print_timings"), "set_print_timings", "get_print_timings");

    ClassDB::bind_method(D_METHOD("set_n_threads", "n_threads"), &LlamaMultimodalConfig::set_n_threads);
    ClassDB::bind_method(D_METHOD("get_n_threads"), &LlamaMultimodalConfig::get_n_threads);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "n_threads"), "set_n_threads", "get_n_threads");

    ClassDB::bind_method(D_METHOD("set_image_min_tokens", "image_min_tokens"),
                         &LlamaMultimodalConfig::set_image_min_tokens);
    ClassDB::bind_method(D_METHOD("get_image_min_tokens"), &LlamaMultimodalConfig::get_image_min_tokens);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "image_min_tokens"), "set_image_min_tokens", "get_image_min_tokens");

    ClassDB::bind_method(D_METHOD("set_image_max_tokens", "image_max_tokens"),
                         &LlamaMultimodalConfig::set_image_max_tokens);
    ClassDB::bind_method(D_METHOD("get_image_max_tokens"), &LlamaMultimodalConfig::get_image_max_tokens);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "image_max_tokens"), "set_image_max_tokens", "get_image_max_tokens");
}

void LlamaMultimodalConfig::set_mmproj_path(const String &p_path) {
    mmproj_path_ = p_path;
}

String LlamaMultimodalConfig::get_mmproj_path() const {
    return mmproj_path_;
}

void LlamaMultimodalConfig::set_media_marker(const String &p_marker) {
    media_marker_ = p_marker.is_empty() ? String("<__media__>") : p_marker;
}

String LlamaMultimodalConfig::get_media_marker() const {
    return media_marker_;
}

void LlamaMultimodalConfig::set_use_gpu(bool p_use_gpu) {
    use_gpu_ = p_use_gpu;
}

bool LlamaMultimodalConfig::get_use_gpu() const {
    return use_gpu_;
}

void LlamaMultimodalConfig::set_print_timings(bool p_print_timings) {
    print_timings_ = p_print_timings;
}

bool LlamaMultimodalConfig::get_print_timings() const {
    return print_timings_;
}

void LlamaMultimodalConfig::set_n_threads(int32_t p_n_threads) {
    n_threads_ = p_n_threads;
}

int32_t LlamaMultimodalConfig::get_n_threads() const {
    return n_threads_;
}

void LlamaMultimodalConfig::set_image_min_tokens(int32_t p_value) {
    image_min_tokens_ = p_value < 0 ? 0 : p_value;
}

int32_t LlamaMultimodalConfig::get_image_min_tokens() const {
    return image_min_tokens_;
}

void LlamaMultimodalConfig::set_image_max_tokens(int32_t p_value) {
    image_max_tokens_ = p_value < 0 ? 0 : p_value;
}

int32_t LlamaMultimodalConfig::get_image_max_tokens() const {
    return image_max_tokens_;
}

} // namespace godot
