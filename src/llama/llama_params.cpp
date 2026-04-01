#include "godot_llama/llama_params.hpp"

// ModelConfig and GenerateOptions are pure data structures.
// This translation unit exists to anchor their vtables/linkage if needed
// and to keep the build system consistent.

namespace godot_llama {
// intentionally empty — config types are header-only data
} // namespace godot_llama
