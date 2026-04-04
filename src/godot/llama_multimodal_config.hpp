#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot {

class LlamaMultimodalConfig : public Resource {
    GDCLASS(LlamaMultimodalConfig, Resource)

public:
    LlamaMultimodalConfig() = default;
    ~LlamaMultimodalConfig() override = default;

    void set_mmproj_path(const String &p_path);
    String get_mmproj_path() const;

    void set_media_marker(const String &p_marker);
    String get_media_marker() const;

    void set_use_gpu(bool p_use_gpu);
    bool get_use_gpu() const;

    void set_print_timings(bool p_print_timings);
    bool get_print_timings() const;

    void set_n_threads(int32_t p_n_threads);
    int32_t get_n_threads() const;

    void set_image_min_tokens(int32_t p_value);
    int32_t get_image_min_tokens() const;

    void set_image_max_tokens(int32_t p_value);
    int32_t get_image_max_tokens() const;

protected:
    static void _bind_methods();

private:
    String mmproj_path_;
    String media_marker_ = "<__media__>";
    bool use_gpu_ = false;
    bool print_timings_ = false;
    int32_t n_threads_ = -1;
    int32_t image_min_tokens_ = 0;
    int32_t image_max_tokens_ = 0;
};

} // namespace godot
