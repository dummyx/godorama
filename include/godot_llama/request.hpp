#pragma once

#include "godot_llama/error.hpp"
#include "godot_llama/llama_params.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace godot_llama {

using RequestId = int32_t;

struct GenerateResult {
    RequestId request_id = 0;
    std::string full_text;
    int32_t tokens_generated = 0;
    double time_ms = 0.0;
    double tokens_per_second = 0.0;
};

struct TokenEvent {
    RequestId request_id = 0;
    std::string text;
    int32_t token_id = 0;
};

struct ErrorEvent {
    RequestId request_id = 0;
    ErrorCode code = ErrorCode::InternalError;
    std::string message;
    std::string details;
};

struct GenerateRequest {
    RequestId id = 0;
    std::string prompt;
    std::vector<MultimodalInput> media_inputs;
    GenerateOptions options;
    std::atomic<bool> cancelled{false};

    void cancel() noexcept { cancelled.store(true, std::memory_order_release); }
    [[nodiscard]] bool is_cancelled() const noexcept { return cancelled.load(std::memory_order_acquire); }
};

// Callback interfaces for delivering results back to the caller.
// These may be called from a worker thread.
struct RequestCallbacks {
    std::function<void(const TokenEvent &)> on_token;
    std::function<void(const GenerateResult &)> on_complete;
    std::function<void(const ErrorEvent &)> on_error;
    std::function<void(RequestId)> on_cancelled;
};

} // namespace godot_llama
