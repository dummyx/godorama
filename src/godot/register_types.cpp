#include "register_types.hpp"

#include "llama_eval_session.hpp"
#include "llama_lora_adapter_config.hpp"
#include "llama_model_config.hpp"
#include "llama_multimodal_config.hpp"
#include "llama_session.hpp"
#include "rag_answer_session.hpp"
#include "rag_corpus.hpp"
#include "rag_corpus_config.hpp"

#include <llama.h>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

#include <gdextension_interface.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>

using namespace godot;

static bool llama_backend_initialized = false;

namespace {

struct LlamaLogState {
    std::mutex mutex;
    std::optional<ggml_log_level> pending_log_level;
};

LlamaLogState &llama_log_state() {
    // Keep the callback state alive until process exit so late llama/ggml teardown
    // logs cannot race static destruction of the mutex.
    static auto *state = new LlamaLogState();
    return *state;
}

ggml_log_level resolve_llama_log_threshold() {
    const char *raw = std::getenv("GODORAMA_LLAMA_LOG_LEVEL");
    if (raw == nullptr || *raw == '\0') {
        return GGML_LOG_LEVEL_WARN;
    }

    std::string value(raw);
    for (char &ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (value == "debug") {
        return GGML_LOG_LEVEL_DEBUG;
    }
    if (value == "info") {
        return GGML_LOG_LEVEL_INFO;
    }
    if (value == "warn" || value == "warning") {
        return GGML_LOG_LEVEL_WARN;
    }
    if (value == "error") {
        return GGML_LOG_LEVEL_ERROR;
    }
    if (value == "none" || value == "silent") {
        return static_cast<ggml_log_level>(GGML_LOG_LEVEL_ERROR + 1);
    }

    return GGML_LOG_LEVEL_WARN;
}

void godorama_llama_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void) user_data;

    static const ggml_log_level threshold = resolve_llama_log_threshold();
    LlamaLogState &state = llama_log_state();

    std::lock_guard<std::mutex> lock(state.mutex);

    if (level == GGML_LOG_LEVEL_CONT) {
        if (!state.pending_log_level.has_value() || state.pending_log_level.value() < threshold) {
            return;
        }
        std::fputs(text, stderr);
        std::fflush(stderr);
        return;
    }

    state.pending_log_level = level;
    if (level < threshold) {
        return;
    }

    std::fputs(text, stderr);
    std::fflush(stderr);
}

const char *ggml_backend_device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "ACCEL";
    }

    return "UNKNOWN";
}

void log_ggml_backend_devices() {
    char buffer[512];
    const size_t device_count = ggml_backend_dev_count();

    std::snprintf(buffer, sizeof(buffer), "godorama: registered %zu ggml backend device(s)\n", device_count);
    godorama_llama_log_callback(GGML_LOG_LEVEL_INFO, buffer, nullptr);

    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const char *name = ggml_backend_dev_name(device);
        const char *description = ggml_backend_dev_description(device);
        const char *type_name = ggml_backend_device_type_name(ggml_backend_dev_type(device));

        std::snprintf(buffer, sizeof(buffer),
                      "godorama: ggml device %zu: name=%s type=%s description=%s\n",
                      i,
                      name != nullptr ? name : "(null)",
                      type_name,
                      description != nullptr ? description : "(null)");
        godorama_llama_log_callback(GGML_LOG_LEVEL_INFO, buffer, nullptr);
    }

    if (!llama_supports_gpu_offload()) {
        godorama_llama_log_callback(GGML_LOG_LEVEL_WARN,
                                    "godorama: no ggml GPU backend device registered; GGUF inference will run on CPU\n",
                                    nullptr);
    }
}

} // namespace

void initialize_godot_llama_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    if (!llama_backend_initialized) {
        llama_log_set(godorama_llama_log_callback, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        log_ggml_backend_devices();
        llama_backend_initialized = true;
    }

    GDREGISTER_CLASS(LlamaLoraAdapterConfig);
    GDREGISTER_CLASS(LlamaMultimodalConfig);
    GDREGISTER_CLASS(LlamaModelConfig);
    GDREGISTER_CLASS(LlamaEvalSession);
    GDREGISTER_CLASS(LlamaSession);
    GDREGISTER_CLASS(RagCorpusConfig);
    GDREGISTER_CLASS(RagCorpus);
    GDREGISTER_CLASS(RagAnswerSession);
}

void uninitialize_godot_llama_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    if (llama_backend_initialized) {
        llama_log_set(nullptr, nullptr);
        // On macOS headless runs we have observed shutdown-time
        // std::system_error("mutex lock failed") crashes inside or after
        // backend teardown. The process is already exiting here, so favor a
        // stable shutdown over aggressively freeing llama's global backend
        // state.
        llama_backend_initialized = false;
    }
}

extern "C" {

GDExtensionBool GDE_EXPORT godot_llama_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address,
                                                    GDExtensionClassLibraryPtr p_library,
                                                    GDExtensionInitialization *r_initialization) {
    GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

    init_obj.register_initializer(initialize_godot_llama_module);
    init_obj.register_terminator(uninitialize_godot_llama_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}
}
