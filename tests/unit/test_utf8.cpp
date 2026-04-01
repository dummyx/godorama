#include <catch2/catch_test_macros.hpp>

#include "godot_llama/utf8.hpp"

#include <string_view>

using namespace godot_llama;

// Helper to convert C++20 char8_t strings to char string_view
static std::string_view sv(const char8_t *s) {
    return {reinterpret_cast<const char *>(s), std::char_traits<char8_t>::length(s)};
}

TEST_CASE("UTF-8 validation", "[utf8]") {
    SECTION("valid ASCII") {
        REQUIRE(utf8::is_valid("hello world"));
        REQUIRE(utf8::is_valid(""));
        REQUIRE(utf8::is_valid("0123456789"));
    }

    SECTION("valid multibyte") {
        REQUIRE(utf8::is_valid(sv(u8"café")));
        REQUIRE(utf8::is_valid(sv(u8"日本語")));
        REQUIRE(utf8::is_valid(sv(u8"🦙")));
        REQUIRE(utf8::is_valid(sv(u8"Ünîcödé")));
    }

    SECTION("invalid sequences") {
        REQUIRE_FALSE(utf8::is_valid("\x80"));
        REQUIRE_FALSE(utf8::is_valid("\xC0\x80")); // overlong
        REQUIRE_FALSE(utf8::is_valid("\xC1\xBF")); // overlong
        REQUIRE_FALSE(utf8::is_valid("\xFE"));
        REQUIRE_FALSE(utf8::is_valid("\xFF"));
        REQUIRE_FALSE(utf8::is_valid("\xED\xA0\x80")); // surrogate
    }

    SECTION("truncated sequences") {
        REQUIRE_FALSE(utf8::is_valid("\xC2"));         // 2-byte truncated
        REQUIRE_FALSE(utf8::is_valid("\xE0\xA0"));     // 3-byte truncated
        REQUIRE_FALSE(utf8::is_valid("\xF0\x90\x80")); // 4-byte truncated
    }
}

TEST_CASE("UTF-8 codepoint counting", "[utf8]") {
    SECTION("ASCII") {
        REQUIRE(utf8::codepoint_count("hello") == 5);
        REQUIRE(utf8::codepoint_count("") == 0);
    }

    SECTION("multibyte characters") {
        REQUIRE(utf8::codepoint_count(sv(u8"café")) == 4);
        REQUIRE(utf8::codepoint_count(sv(u8"🦙")) == 1);
        REQUIRE(utf8::codepoint_count(sv(u8"日本語")) == 3);
    }
}
