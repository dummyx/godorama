#include <catch2/catch_test_macros.hpp>

#include "godot_llama/llama_position_layout.hpp"

using namespace godot_llama;

TEST_CASE("Position layout defaults to linear positions for standard models", "[position_layout]") {
    std::vector<int32_t> normalized;
    const auto err = normalize_position_layout({}, 4, 1, normalized);

    REQUIRE_FALSE(err);
    REQUIRE(normalized == std::vector<int32_t>({0, 1, 2, 3}));
}

TEST_CASE("Position layout expands default positions for 3-component M-RoPE", "[position_layout]") {
    std::vector<int32_t> normalized;
    const auto err = normalize_position_layout({}, 3, 3, normalized);

    REQUIRE_FALSE(err);
    REQUIRE(normalized == std::vector<int32_t>({0, 1, 2, 0, 1, 2, 0, 1, 2}));
}

TEST_CASE("Position layout expands base positions for 3-component M-RoPE", "[position_layout]") {
    std::vector<int32_t> normalized;
    const std::vector<int32_t> base_positions = {5, 6, 7};
    const auto err = normalize_position_layout(base_positions, 3, 3, normalized);

    REQUIRE_FALSE(err);
    REQUIRE(normalized == std::vector<int32_t>({5, 6, 7, 5, 6, 7, 5, 6, 7}));
}

TEST_CASE("Position layout expands base positions for 4-component audio M-RoPE", "[position_layout]") {
    std::vector<int32_t> normalized;
    const std::vector<int32_t> base_positions = {5, 6, 7};
    const auto err = normalize_position_layout(base_positions, 3, 4, normalized);

    REQUIRE_FALSE(err);
    REQUIRE(normalized == std::vector<int32_t>({5, 6, 7, 5, 6, 7, 5, 6, 7, 0, 0, 0}));
}

TEST_CASE("Position layout rejects mismatched shapes", "[position_layout]") {
    std::vector<int32_t> normalized;
    const std::vector<int32_t> bad_positions = {0, 1};
    const auto err = normalize_position_layout(bad_positions, 3, 1, normalized);

    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::InvalidParameter);
}
