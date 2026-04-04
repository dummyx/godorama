#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

class LlamaLoraAdapterConfig : public Resource {
    GDCLASS(LlamaLoraAdapterConfig, Resource)

public:
    LlamaLoraAdapterConfig() = default;
    ~LlamaLoraAdapterConfig() override = default;

    void set_adapter_path(const String &p_path);
    String get_adapter_path() const;

    void set_scale(double p_scale);
    double get_scale() const;

protected:
    static void _bind_methods();

private:
    String adapter_path_;
    double scale_ = 1.0;
};

} // namespace godot
