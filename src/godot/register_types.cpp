#include "register_types.hpp"

#include "llama_eval_session.hpp"
#include "llama_model_config.hpp"
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

std::mutex s_llama_log_mutex;
std::optional<ggml_log_level> s_pending_log_level;

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

    std::lock_guard<std::mutex> lock(s_llama_log_mutex);

    if (level == GGML_LOG_LEVEL_CONT) {
        if (!s_pending_log_level.has_value() || s_pending_log_level.value() < threshold) {
            return;
        }
        std::fputs(text, stderr);
        std::fflush(stderr);
        return;
    }

    s_pending_log_level = level;
    if (level < threshold) {
        return;
    }

    std::fputs(text, stderr);
    std::fflush(stderr);
}

} // namespace

void initialize_godot_llama_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    if (!llama_backend_initialized) {
        llama_backend_init();
        llama_log_set(godorama_llama_log_callback, nullptr);
        llama_backend_initialized = true;
    }

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
        llama_backend_free();
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
