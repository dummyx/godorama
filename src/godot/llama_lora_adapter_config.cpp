#include "llama_lora_adapter_config.hpp"

#include <godot_cpp/core/class_db.hpp>

namespace godot {

void LlamaLoraAdapterConfig::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_adapter_path", "path"), &LlamaLoraAdapterConfig::set_adapter_path);
    ClassDB::bind_method(D_METHOD("get_adapter_path"), &LlamaLoraAdapterConfig::get_adapter_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "adapter_path", PROPERTY_HINT_FILE, "*.gguf"), "set_adapter_path",
                 "get_adapter_path");

    ClassDB::bind_method(D_METHOD("set_scale", "scale"), &LlamaLoraAdapterConfig::set_scale);
    ClassDB::bind_method(D_METHOD("get_scale"), &LlamaLoraAdapterConfig::get_scale);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "scale"), "set_scale", "get_scale");
}

void LlamaLoraAdapterConfig::set_adapter_path(const String &p_path) {
    adapter_path_ = p_path;
}

String LlamaLoraAdapterConfig::get_adapter_path() const {
    return adapter_path_;
}

void LlamaLoraAdapterConfig::set_scale(double p_scale) {
    scale_ = p_scale;
}

double LlamaLoraAdapterConfig::get_scale() const {
    return scale_;
}

} // namespace godot
