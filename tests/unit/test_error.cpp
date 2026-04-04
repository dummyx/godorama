#include <catch2/catch_test_macros.hpp>

#include "godot_llama/error.hpp"

using namespace godot_llama;

// ---------------------------------------------------------------------------
// Error::make_ok
// ---------------------------------------------------------------------------

TEST_CASE("Error::make_ok produces a success", "[error]") {
    auto err = Error::make_ok();
    REQUIRE(err.ok());
    REQUIRE_FALSE(static_cast<bool>(err));
    REQUIRE(err.code == ErrorCode::Ok);
    REQUIRE(err.message.empty());
    REQUIRE(err.context.empty());
}

// ---------------------------------------------------------------------------
// Error::make
// ---------------------------------------------------------------------------

TEST_CASE("Error::make with message and context", "[error]") {
    auto err = Error::make(ErrorCode::ModelLoadFailed, "bad model", "path=/foo.gguf");
    REQUIRE_FALSE(err.ok());
    REQUIRE(static_cast<bool>(err));
    REQUIRE(err.code == ErrorCode::ModelLoadFailed);
    REQUIRE(err.message == "bad model");
    REQUIRE(err.context == "path=/foo.gguf");
}

TEST_CASE("Error::make with message only", "[error]") {
    auto err = Error::make(ErrorCode::InvalidParameter, "param is wrong");
    REQUIRE(err.code == ErrorCode::InvalidParameter);
    REQUIRE(err.message == "param is wrong");
    REQUIRE(err.context.empty());
}

TEST_CASE("Error::make preserves long messages", "[error]") {
    std::string long_msg(1024, 'X');
    std::string long_ctx(512, 'C');
    auto err = Error::make(ErrorCode::InternalError, long_msg, long_ctx);
    REQUIRE(err.message.size() == 1024);
    REQUIRE(err.context.size() == 512);
    REQUIRE(err.message.front() == 'X');
    REQUIRE(err.context.front() == 'C');
}

// ---------------------------------------------------------------------------
// Error copy
// ---------------------------------------------------------------------------

TEST_CASE("Error is copyable", "[error]") {
    auto original = Error::make(ErrorCode::TokenizeFailed, "tokenize error", "detail");
    auto copy = original;

    REQUIRE(copy.code == ErrorCode::TokenizeFailed);
    REQUIRE(copy.message == "tokenize error");
    REQUIRE(copy.context == "detail");

    // Mutating copy does not affect original
    copy.message = "changed";
    REQUIRE(original.message == "tokenize error");
    REQUIRE(copy.message == "changed");
}

// ---------------------------------------------------------------------------
// Error::operator bool consistency
// ---------------------------------------------------------------------------

TEST_CASE("Error operator bool is inverse of ok", "[error]") {
    auto ok_err = Error::make_ok();
    auto fail_err = Error::make(ErrorCode::DecodeFailed, "decode error");

    REQUIRE(ok_err.ok() == true);
    REQUIRE(static_cast<bool>(ok_err) == false);
    REQUIRE(fail_err.ok() == false);
    REQUIRE(static_cast<bool>(fail_err) == true);
}

// ---------------------------------------------------------------------------
// ErrorCode values
// ---------------------------------------------------------------------------

TEST_CASE("ErrorCode underlying values are stable", "[error]") {
    REQUIRE(static_cast<int32_t>(ErrorCode::Ok) == 0);
    REQUIRE(static_cast<int32_t>(ErrorCode::InvalidPath) != 0);
    REQUIRE(static_cast<int32_t>(ErrorCode::Cancelled) != 0);
    REQUIRE(static_cast<int32_t>(ErrorCode::InternalError) != 0);
}

// ---------------------------------------------------------------------------
// error_code_name
// ---------------------------------------------------------------------------

TEST_CASE("error_code_name returns non-empty for all known codes", "[error]") {
    // Exhaustively test every enumerator
    const auto codes = {
            ErrorCode::Ok,
            ErrorCode::InvalidPath,
            ErrorCode::ModelLoadFailed,
            ErrorCode::ContextCreateFailed,
            ErrorCode::NotOpen,
            ErrorCode::AlreadyOpen,
            ErrorCode::InvalidParameter,
            ErrorCode::TokenizeFailed,
            ErrorCode::DecodeFailed,
            ErrorCode::Cancelled,
            ErrorCode::InternalError,
            ErrorCode::InvalidUtf8,
            ErrorCode::UnsupportedFormat,
            ErrorCode::StorageOpenFailed,
            ErrorCode::StorageMigrationFailed,
            ErrorCode::StorageCorrupt,
            ErrorCode::CapabilityUnavailable,
            ErrorCode::EmbeddingsUnavailable,
            ErrorCode::RerankerUnavailable,
            ErrorCode::SchemaMismatch,
            ErrorCode::RetrievalFailed,
            ErrorCode::IngestionFailed,
            ErrorCode::MetadataTooLarge,
            ErrorCode::StaleEmbeddings,
            ErrorCode::QueueFull,
            ErrorCode::BudgetExceeded,
    };

    for (const auto &code : codes) {
        auto name = error_code_name(code);
        REQUIRE_FALSE(name.empty());
        REQUIRE(name != "UNKNOWN");
    }
}

TEST_CASE("error_code_name returns UNKNOWN for out-of-range value", "[error]") {
    // Cast an invalid value — this is UB in a strict sense, but the switch-default
    // is designed to return "UNKNOWN" defensively.
    auto name = error_code_name(static_cast<ErrorCode>(99999));
    REQUIRE(name == "UNKNOWN");
}

TEST_CASE("error_code_name specific values", "[error]") {
    REQUIRE(error_code_name(ErrorCode::Ok) == "OK");
    REQUIRE(error_code_name(ErrorCode::InvalidPath) == "INVALID_PATH");
    REQUIRE(error_code_name(ErrorCode::ModelLoadFailed) == "MODEL_LOAD_FAILED");
    REQUIRE(error_code_name(ErrorCode::ContextCreateFailed) == "CONTEXT_CREATE_FAILED");
    REQUIRE(error_code_name(ErrorCode::NotOpen) == "NOT_OPEN");
    REQUIRE(error_code_name(ErrorCode::AlreadyOpen) == "ALREADY_OPEN");
    REQUIRE(error_code_name(ErrorCode::InvalidParameter) == "INVALID_PARAMETER");
    REQUIRE(error_code_name(ErrorCode::TokenizeFailed) == "TOKENIZE_FAILED");
    REQUIRE(error_code_name(ErrorCode::DecodeFailed) == "DECODE_FAILED");
    REQUIRE(error_code_name(ErrorCode::Cancelled) == "CANCELLED");
    REQUIRE(error_code_name(ErrorCode::InternalError) == "INTERNAL_ERROR");
    REQUIRE(error_code_name(ErrorCode::InvalidUtf8) == "INVALID_UTF8");
    REQUIRE(error_code_name(ErrorCode::UnsupportedFormat) == "UNSUPPORTED_FORMAT");
    REQUIRE(error_code_name(ErrorCode::CapabilityUnavailable) == "CAPABILITY_UNAVAILABLE");
    REQUIRE(error_code_name(ErrorCode::EmbeddingsUnavailable) == "EMBEDDINGS_UNAVAILABLE");
}

TEST_CASE("error_code_name returns string_view", "[error]") {
    // Confirm that the return type is cheap to inspect (no allocation)
    auto name = error_code_name(ErrorCode::Ok);
    static_assert(std::is_same_v<decltype(name), std::string_view>);
    REQUIRE(name == "OK");
}

// ---------------------------------------------------------------------------
// Error equality comparison
// ---------------------------------------------------------------------------

TEST_CASE("Error code comparison", "[error]") {
    auto err1 = Error::make(ErrorCode::InvalidPath, "msg1");
    auto err2 = Error::make(ErrorCode::InvalidPath, "msg2");
    auto err3 = Error::make(ErrorCode::ModelLoadFailed, "msg1");

    REQUIRE(err1.code == err2.code);
    REQUIRE(err1.code != err3.code);
}

// ---------------------------------------------------------------------------
// Error context is optional
// ---------------------------------------------------------------------------

TEST_CASE("Error without context", "[error]") {
    auto err = Error::make(ErrorCode::NotOpen, "not open");
    REQUIRE(err.context.empty());
    REQUIRE(err.code == ErrorCode::NotOpen);
}

TEST_CASE("Error with empty context", "[error]") {
    auto err = Error::make(ErrorCode::NotOpen, "not open", "");
    REQUIRE(err.context.empty());
}

TEST_CASE("Error with multi-line context", "[error]") {
    auto err = Error::make(ErrorCode::DecodeFailed, "decode failed", "line1\nline2\nline3");
    REQUIRE(err.context.find('\n') != std::string::npos);
}
