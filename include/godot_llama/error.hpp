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
    InvalidUtf8,
    UnsupportedFormat,
    StorageOpenFailed,
    StorageMigrationFailed,
    StorageCorrupt,
    CapabilityUnavailable,
    EmbeddingsUnavailable,
    RerankerUnavailable,
    SchemaMismatch,
    RetrievalFailed,
    IngestionFailed,
    MetadataTooLarge,
    StaleEmbeddings,
    QueueFull,
    BudgetExceeded,
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
    case ErrorCode::InvalidUtf8:
        return "INVALID_UTF8";
    case ErrorCode::UnsupportedFormat:
        return "UNSUPPORTED_FORMAT";
    case ErrorCode::StorageOpenFailed:
        return "STORAGE_OPEN_FAILED";
    case ErrorCode::StorageMigrationFailed:
        return "STORAGE_MIGRATION_FAILED";
    case ErrorCode::StorageCorrupt:
        return "STORAGE_CORRUPT";
    case ErrorCode::CapabilityUnavailable:
        return "CAPABILITY_UNAVAILABLE";
    case ErrorCode::EmbeddingsUnavailable:
        return "EMBEDDINGS_UNAVAILABLE";
    case ErrorCode::RerankerUnavailable:
        return "RERANKER_UNAVAILABLE";
    case ErrorCode::SchemaMismatch:
        return "SCHEMA_MISMATCH";
    case ErrorCode::RetrievalFailed:
        return "RETRIEVAL_FAILED";
    case ErrorCode::IngestionFailed:
        return "INGESTION_FAILED";
    case ErrorCode::MetadataTooLarge:
        return "METADATA_TOO_LARGE";
    case ErrorCode::StaleEmbeddings:
        return "STALE_EMBEDDINGS";
    case ErrorCode::QueueFull:
        return "QUEUE_FULL";
    case ErrorCode::BudgetExceeded:
        return "BUDGET_EXCEEDED";
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
