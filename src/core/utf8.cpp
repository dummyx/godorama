#include "godot_llama/utf8.hpp"

namespace godot_llama {
namespace utf8 {

bool is_valid(std::string_view s) noexcept {
    const auto *p = reinterpret_cast<const uint8_t *>(s.data());
    const auto *end = p + s.size();

    while (p < end) {
        if (*p < 0x80) {
            ++p;
        } else if ((*p & 0xE0) == 0xC0) {
            if (end - p < 2) {
                return false;
            }
            if ((p[1] & 0xC0) != 0x80) {
                return false;
            }
            if (*p < 0xC2) {
                return false; // overlong
            }
            p += 2;
        } else if ((*p & 0xF0) == 0xE0) {
            if (end - p < 3) {
                return false;
            }
            if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) {
                return false;
            }
            uint32_t cp = (static_cast<uint32_t>(p[0] & 0x0F) << 12) | (static_cast<uint32_t>(p[1] & 0x3F) << 6) |
                          static_cast<uint32_t>(p[2] & 0x3F);
            if (cp < 0x0800 || (cp >= 0xD800 && cp <= 0xDFFF)) {
                return false;
            }
            p += 3;
        } else if ((*p & 0xF8) == 0xF0) {
            if (end - p < 4) {
                return false;
            }
            if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) {
                return false;
            }
            uint32_t cp = (static_cast<uint32_t>(p[0] & 0x07) << 18) | (static_cast<uint32_t>(p[1] & 0x3F) << 12) |
                          (static_cast<uint32_t>(p[2] & 0x3F) << 6) | static_cast<uint32_t>(p[3] & 0x3F);
            if (cp < 0x10000 || cp > 0x10FFFF) {
                return false;
            }
            p += 4;
        } else {
            return false;
        }
    }

    return true;
}

int32_t codepoint_count(std::string_view s) noexcept {
    int32_t count = 0;
    const auto *p = reinterpret_cast<const uint8_t *>(s.data());
    const auto *end = p + s.size();

    while (p < end) {
        if (*p < 0x80) {
            ++p;
        } else if ((*p & 0xE0) == 0xC0) {
            p += 2;
        } else if ((*p & 0xF0) == 0xE0) {
            p += 3;
        } else if ((*p & 0xF8) == 0xF0) {
            p += 4;
        } else {
            ++p; // invalid byte, skip
        }
        ++count;
    }

    return count;
}

} // namespace utf8
} // namespace godot_llama
