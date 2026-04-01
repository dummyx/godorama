#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace godot_llama {

enum class ErrorCode : int32_t {
    Ok = 0,
    InvalidPath,
    ModelLoadFailed,
    ContextCreateFailed,
    NotOpen,
    AlreadyOpen,
    InvalidParameter,
    TokenizeFailed,
    DecodeFailed,
    Cancelled,
    InternalError,
};

[[nodiscard]] constexpr std::string_view error_code_name(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::Ok:
        return "OK";
    case ErrorCode::InvalidPath:
        return "INVALID_PATH";
    case ErrorCode::ModelLoadFailed:
        return "MODEL_LOAD_FAILED";
    case ErrorCode::ContextCreateFailed:
        return "CONTEXT_CREATE_FAILED";
    case ErrorCode::NotOpen:
        return "NOT_OPEN";
    case ErrorCode::AlreadyOpen:
        return "ALREADY_OPEN";
    case ErrorCode::InvalidParameter:
        return "INVALID_PARAMETER";
    case ErrorCode::TokenizeFailed:
        return "TOKENIZE_FAILED";
    case ErrorCode::DecodeFailed:
        return "DECODE_FAILED";
    case ErrorCode::Cancelled:
        return "CANCELLED";
    case ErrorCode::InternalError:
        return "INTERNAL_ERROR";
    }
    return "UNKNOWN";
}

struct Error {
    ErrorCode code = ErrorCode::Ok;
    std::string message;
    std::string context;

    [[nodiscard]] bool ok() const noexcept { return code == ErrorCode::Ok; }
    [[nodiscard]] explicit operator bool() const noexcept { return !ok(); }

    static Error make_ok() noexcept { return {ErrorCode::Ok, {}, {}}; }

    static Error make(ErrorCode c, std::string msg, std::string ctx = {}) noexcept {
        return {c, std::move(msg), std::move(ctx)};
    }
};

} // namespace godot_llama
