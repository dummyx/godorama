#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace godot_llama {
namespace utf8 {

// Validate that a byte sequence is valid UTF-8
[[nodiscard]] bool is_valid(std::string_view s) noexcept;

// Count the number of Unicode code points in a UTF-8 string
[[nodiscard]] int32_t codepoint_count(std::string_view s) noexcept;

} // namespace utf8
} // namespace godot_llama
