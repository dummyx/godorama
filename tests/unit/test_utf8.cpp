#include <catch2/catch_test_macros.hpp>

#include "godot_llama/utf8.hpp"

#include <string>
#include <string_view>

using namespace godot_llama;

// Helper to convert C++20 char8_t strings to char string_view
static std::string_view sv(const char8_t *s) {
    return {reinterpret_cast<const char *>(s), std::char_traits<char8_t>::length(s)};
}

// ---------------------------------------------------------------------------
// is_valid
// ---------------------------------------------------------------------------

TEST_CASE("UTF-8 validation: empty string", "[utf8]") {
    REQUIRE(utf8::is_valid(""));
}

TEST_CASE("UTF-8 validation: pure ASCII", "[utf8]") {
    REQUIRE(utf8::is_valid("hello world"));
    REQUIRE(utf8::is_valid("0123456789"));
    REQUIRE(utf8::is_valid("ABCDEFGHIJKLMNOPQRSTUVWXYZ"));
    REQUIRE(utf8::is_valid("abcdefghijklmnopqrstuvwxyz"));
    REQUIRE(utf8::is_valid("!@#$%^&*()_+-=[]{}|;':\",./<>?"));
    REQUIRE(utf8::is_valid(" "));
    REQUIRE(utf8::is_valid("\t\n\r"));
}

TEST_CASE("UTF-8 validation: single-byte range", "[utf8]") {
    // All valid single-byte characters (0x00-0x7F)
    for (unsigned char c = 0; c <= 0x7F; ++c) {
        char buf[2] = {static_cast<char>(c), '\0'};
        REQUIRE(utf8::is_valid(buf));
    }
}

TEST_CASE("UTF-8 validation: valid 2-byte sequences", "[utf8]") {
    // 2-byte range: U+0080 to U+07FF
    REQUIRE(utf8::is_valid(sv(u8"café")));
    REQUIRE(utf8::is_valid(sv(u8"Ünîcödé")));
    REQUIRE(utf8::is_valid(sv(u8"Ñ")));
    REQUIRE(utf8::is_valid(sv(u8"€"))); // U+20AC (3-byte actually, but confirms multibyte)
}

TEST_CASE("UTF-8 validation: valid 3-byte sequences", "[utf8]") {
    REQUIRE(utf8::is_valid(sv(u8"日本語")));
    REQUIRE(utf8::is_valid(sv(u8"中文")));
    REQUIRE(utf8::is_valid(sv(u8"한글")));
    REQUIRE(utf8::is_valid(sv(u8"Привет")));
    REQUIRE(utf8::is_valid(sv(u8"العربية")));
}

TEST_CASE("UTF-8 validation: valid 4-byte sequences", "[utf8]") {
    REQUIRE(utf8::is_valid(sv(u8"🦙")));
    REQUIRE(utf8::is_valid(sv(u8"🎉")));
    REQUIRE(utf8::is_valid(sv(u8"👨‍👩‍👧‍👦"))); // ZWJ sequence
    REQUIRE(utf8::is_valid(sv(u8"𝄞"))); // U+1D11E Musical Symbol G Clef
}

TEST_CASE("UTF-8 validation: mixed ASCII and multibyte", "[utf8]") {
    REQUIRE(utf8::is_valid(sv(u8"Hello, 世界!")));
    REQUIRE(utf8::is_valid(sv(u8"Café ☕ time")));
    REQUIRE(utf8::is_valid(sv(u8"Test 🦙 llama.cpp 🎉")));
}

TEST_CASE("UTF-8 validation: invalid continuation byte", "[utf8]") {
    REQUIRE_FALSE(utf8::is_valid("\x80")); // bare continuation byte
    REQUIRE_FALSE(utf8::is_valid("\xBF")); // bare continuation byte
    REQUIRE_FALSE(utf8::is_valid("hello\x80world"));
}

TEST_CASE("UTF-8 validation: invalid start bytes", "[utf8]") {
    REQUIRE_FALSE(utf8::is_valid("\xFE"));
    REQUIRE_FALSE(utf8::is_valid("\xFF"));
    REQUIRE_FALSE(utf8::is_valid("\xFE\xFF"));
}

TEST_CASE("UTF-8 validation: overlong encodings", "[utf8]") {
    REQUIRE_FALSE(utf8::is_valid("\xC0\x80")); // overlong NUL
    REQUIRE_FALSE(utf8::is_valid("\xC1\xBF")); // overlong 0x7F
    REQUIRE_FALSE(utf8::is_valid("\xE0\x80\x80")); // overlong 0x00
    REQUIRE_FALSE(utf8::is_valid("\xF0\x80\x80\x80")); // overlong 0x00
}

TEST_CASE("UTF-8 validation: truncated sequences", "[utf8]") {
    REQUIRE_FALSE(utf8::is_valid("\xC2"));         // 2-byte truncated
    REQUIRE_FALSE(utf8::is_valid("\xE0\xA0"));     // 3-byte truncated
    REQUIRE_FALSE(utf8::is_valid("\xF0\x90\x80")); // 4-byte truncated
}

TEST_CASE("UTF-8 validation: surrogate range", "[utf8]") {
    REQUIRE_FALSE(utf8::is_valid("\xED\xA0\x80")); // U+D800 (high surrogate)
    REQUIRE_FALSE(utf8::is_valid("\xED\xAF\xBF")); // U+DBFF (high surrogate)
    REQUIRE_FALSE(utf8::is_valid("\xED\xB0\x80")); // U+DC00 (low surrogate)
    REQUIRE_FALSE(utf8::is_valid("\xED\xBF\xBF")); // U+DFFF (low surrogate)
}

TEST_CASE("UTF-8 validation: wrong continuation byte", "[utf8]") {
    REQUIRE_FALSE(utf8::is_valid("\xC2\x00"));     // NUL instead of continuation
    REQUIRE_FALSE(utf8::is_valid("\xC2\x41"));     // 'A' instead of continuation
    REQUIRE_FALSE(utf8::is_valid("\xE0\xA0\x00")); // NUL in 3-byte
}

TEST_CASE("UTF-8 validation: sequence at end of longer string", "[utf8]") {
    // Valid prefix but invalid trailing bytes
    REQUIRE_FALSE(utf8::is_valid("abc\xC2"));      // truncated at end
    REQUIRE_FALSE(utf8::is_valid("abc\xE0\xA0"));  // truncated at end
    REQUIRE(utf8::is_valid("abc\xC3\xA9"));        // valid 'é' at end
}

TEST_CASE("UTF-8 validation: long valid strings", "[utf8]") {
    std::string long_ascii(10000, 'A');
    REQUIRE(utf8::is_valid(long_ascii));

    std::string long_mixed;
    for (int i = 0; i < 1000; ++i) {
        long_mixed += "Hello ";
        long_mixed += reinterpret_cast<const char *>(u8"世界");
    }
    REQUIRE(utf8::is_valid(long_mixed));
}

// ---------------------------------------------------------------------------
// codepoint_count
// ---------------------------------------------------------------------------

TEST_CASE("UTF-8 codepoint counting: empty string", "[utf8]") {
    REQUIRE(utf8::codepoint_count("") == 0);
}

TEST_CASE("UTF-8 codepoint counting: pure ASCII", "[utf8]") {
    REQUIRE(utf8::codepoint_count("hello") == 5);
    REQUIRE(utf8::codepoint_count("a") == 1);
    REQUIRE(utf8::codepoint_count(" ") == 1);
    REQUIRE(utf8::codepoint_count("abcde") == 5);
}

TEST_CASE("UTF-8 codepoint counting: 2-byte characters", "[utf8]") {
    REQUIRE(utf8::codepoint_count(sv(u8"café")) == 4);
    REQUIRE(utf8::codepoint_count(sv(u8"é")) == 1);
    REQUIRE(utf8::codepoint_count(sv(u8"Ñ")) == 1);
}

TEST_CASE("UTF-8 codepoint counting: 3-byte characters", "[utf8]") {
    REQUIRE(utf8::codepoint_count(sv(u8"日本語")) == 3);
    REQUIRE(utf8::codepoint_count(sv(u8"中")) == 1);
    REQUIRE(utf8::codepoint_count(sv(u8"Привет")) == 6);
}

TEST_CASE("UTF-8 codepoint counting: 4-byte characters", "[utf8]") {
    REQUIRE(utf8::codepoint_count(sv(u8"🦙")) == 1);
    REQUIRE(utf8::codepoint_count(sv(u8"🎉🚀✨")) == 3);
    REQUIRE(utf8::codepoint_count(sv(u8"𝄞")) == 1);
}

TEST_CASE("UTF-8 codepoint counting: mixed content", "[utf8]") {
    REQUIRE(utf8::codepoint_count(sv(u8"Hello, 世界!")) == 10); // H,e,l,l,o,',',' ',世,界,!
    REQUIRE(utf8::codepoint_count(sv(u8"2 🦙")) == 3);         // '2', ' ', '🦙'
    REQUIRE(utf8::codepoint_count(sv(u8"aéb")) == 3);          // a, é, b
}

TEST_CASE("UTF-8 codepoint counting: string with newlines", "[utf8]") {
    REQUIRE(utf8::codepoint_count("line1\nline2") == 11);
    REQUIRE(utf8::codepoint_count("\n\n\n") == 3);
}

TEST_CASE("UTF-8 codepoint counting: repeated characters", "[utf8]") {
    std::string repeated(100, 'x');
    REQUIRE(utf8::codepoint_count(repeated) == 100);

    // 100 repetitions of a 3-byte char
    std::string u8_repeated;
    for (int i = 0; i < 100; ++i) {
        u8_repeated += reinterpret_cast<const char *>(u8"日");
    }
    REQUIRE(utf8::codepoint_count(u8_repeated) == 100);
}

TEST_CASE("UTF-8 codepoint counting: byte count differs from codepoint count", "[utf8]") {
    std::string_view text = reinterpret_cast<const char *>(u8"日本語");
    REQUIRE(text.size() == 9); // 3 chars × 3 bytes each
    REQUIRE(utf8::codepoint_count(text) == 3);

    std::string_view emoji = reinterpret_cast<const char *>(u8"🦙");
    REQUIRE(emoji.size() == 4); // 4 bytes for one codepoint
    REQUIRE(utf8::codepoint_count(emoji) == 1);
}

TEST_CASE("UTF-8 codepoint counting: string_view vs string", "[utf8]") {
    std::string s = "hello";
    std::string_view sv = s;
    REQUIRE(utf8::codepoint_count(s) == utf8::codepoint_count(sv));
    REQUIRE(utf8::codepoint_count(s) == 5);
}

// ---------------------------------------------------------------------------
// Integration: is_valid and codepoint_count consistency
// ---------------------------------------------------------------------------

TEST_CASE("UTF-8 valid strings have correct codepoint counts", "[utf8]") {
    auto check = [](std::string_view text, int32_t expected) {
        REQUIRE(utf8::is_valid(text));
        REQUIRE(utf8::codepoint_count(text) == expected);
    };

    check("", 0);
    check("a", 1);
    check("abc", 3);
    check(reinterpret_cast<const char *>(u8"café"), 4);
    check(reinterpret_cast<const char *>(u8"🦙"), 1);
    check(reinterpret_cast<const char *>(u8"Hello 世界"), 8);
}
