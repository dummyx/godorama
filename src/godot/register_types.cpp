#include "register_types.hpp"

#include "llama_model_config.hpp"
#include "llama_session.hpp"

#include <llama.h>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

#include <gdextension_interface.h>

using namespace godot;

static bool llama_backend_initialized = false;

void initialize_godot_llama_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    if (!llama_backend_initialized) {
        llama_backend_init();
        llama_backend_initialized = true;
    }

    GDREGISTER_CLASS(LlamaModelConfig);
    GDREGISTER_CLASS(LlamaSession);
}

void uninitialize_godot_llama_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    if (llama_backend_initialized) {
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
