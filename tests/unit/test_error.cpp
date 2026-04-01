#include <catch2/catch_test_macros.hpp>

#include "godot_llama/error.hpp"

using namespace godot_llama;

TEST_CASE("Error construction", "[error]") {
    SECTION("make_ok produces ok error") {
        auto err = Error::make_ok();
        REQUIRE(err.ok());
        REQUIRE_FALSE(static_cast<bool>(err));
        REQUIRE(err.code == ErrorCode::Ok);
    }

    SECTION("make produces error with message") {
        auto err = Error::make(ErrorCode::ModelLoadFailed, "bad model", "path=/foo");
        REQUIRE_FALSE(err.ok());
        REQUIRE(static_cast<bool>(err));
        REQUIRE(err.code == ErrorCode::ModelLoadFailed);
        REQUIRE(err.message == "bad model");
        REQUIRE(err.context == "path=/foo");
    }
}

TEST_CASE("Error code names", "[error]") {
    REQUIRE(error_code_name(ErrorCode::Ok) == "OK");
    REQUIRE(error_code_name(ErrorCode::InvalidPath) == "INVALID_PATH");
    REQUIRE(error_code_name(ErrorCode::ModelLoadFailed) == "MODEL_LOAD_FAILED");
    REQUIRE(error_code_name(ErrorCode::Cancelled) == "CANCELLED");
}
