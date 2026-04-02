#pragma once

#include "godot_llama/error.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace godot_llama {

[[nodiscard]] Error normalize_position_layout(std::span<const int32_t> positions, int32_t n_tokens,
                                              int32_t n_pos_per_embd, std::vector<int32_t> &out);

} // namespace godot_llama
