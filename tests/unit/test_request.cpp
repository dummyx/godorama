#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "godot_llama/request.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace godot_llama;

// ---------------------------------------------------------------------------
// GenerateRequest
// ---------------------------------------------------------------------------

TEST_CASE("GenerateRequest defaults", "[request]") {
    GenerateRequest req;
    REQUIRE(req.id == 0);
    REQUIRE(req.prompt.empty());
    REQUIRE(req.media_inputs.empty());
    REQUIRE_FALSE(req.is_cancelled());
}

TEST_CASE("GenerateRequest field assignment", "[request]") {
    GenerateRequest req;
    req.id = 42;
    req.prompt = "Hello, world!";

    REQUIRE(req.id == 42);
    REQUIRE(req.prompt == "Hello, world!");
}

TEST_CASE("GenerateRequest cancel sets flag", "[request]") {
    GenerateRequest req;
    REQUIRE_FALSE(req.is_cancelled());

    req.cancel();
    REQUIRE(req.is_cancelled());
}

TEST_CASE("GenerateRequest cancel is idempotent", "[request]") {
    GenerateRequest req;
    req.cancel();
    REQUIRE(req.is_cancelled());
    req.cancel();
    REQUIRE(req.is_cancelled());
}

TEST_CASE("GenerateRequest cancel is visible across threads", "[request]") {
    GenerateRequest req;
    std::atomic<bool> observed{false};

    std::mutex mtx;
    std::condition_variable cv;
    bool started = false;

    std::jthread waiter([&](std::stop_token) {
        {
            std::unique_lock lock(mtx);
            started = true;
            cv.notify_one();
        }
        // Spin until cancelled
        for (int i = 0; i < 10000; ++i) {
            if (req.is_cancelled()) {
                observed.store(true);
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Wait for the thread to start
    {
        std::unique_lock lock(mtx);
        cv.wait(lock, [&] { return started; });
    }

    // Give the thread a moment to start spinning
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    REQUIRE_FALSE(observed.load());

    req.cancel();

    // Thread should observe the cancellation
    waiter.join();
    REQUIRE(observed.load());
}

TEST_CASE("GenerateRequest with media inputs", "[request]") {
    GenerateRequest req;
    req.prompt = "<__media__> Describe this image.";

    MultimodalInput img;
    img.type = MultimodalInputType::Image;
    img.path = "/test/image.jpg";
    req.media_inputs.push_back(std::move(img));

    REQUIRE(req.media_inputs.size() == 1);
    REQUIRE(req.media_inputs[0].type == MultimodalInputType::Image);
    REQUIRE(req.media_inputs[0].path == "/test/image.jpg");
}

TEST_CASE("GenerateRequest with multiple media inputs", "[request]") {
    GenerateRequest req;

    MultimodalInput img;
    img.type = MultimodalInputType::Image;
    img.path = "/test/photo.png";
    req.media_inputs.push_back(std::move(img));

    MultimodalInput audio;
    audio.type = MultimodalInputType::Audio;
    audio.path = "/test/speech.wav";
    req.media_inputs.push_back(std::move(audio));

    REQUIRE(req.media_inputs.size() == 2);
    REQUIRE(req.media_inputs[0].type == MultimodalInputType::Image);
    REQUIRE(req.media_inputs[1].type == MultimodalInputType::Audio);
}

TEST_CASE("GenerateRequest options are stored", "[request]") {
    GenerateRequest req;
    req.options.max_tokens = 500;
    req.options.temperature = 0.5f;
    req.options.seed_override = 42u;
    req.options.stop = {"###", "\n"};

    REQUIRE(req.options.max_tokens == 500);
    REQUIRE(req.options.temperature == 0.5f);
    REQUIRE(req.options.seed_override.has_value());
    REQUIRE(req.options.stop.size() == 2);
}

TEST_CASE("GenerateRequest moved-from media_inputs is empty", "[request]") {
    GenerateRequest req;
    req.media_inputs.push_back({MultimodalInputType::Image, "/test.jpg", "", {}});

    std::vector<MultimodalInput> stolen = std::move(req.media_inputs);
    REQUIRE(stolen.size() == 1);
    // After move, the vector is in a valid-but-unspecified state.
    // In practice, most implementations leave it empty.
}

// ---------------------------------------------------------------------------
// GenerateResult
// ---------------------------------------------------------------------------

TEST_CASE("GenerateResult defaults", "[request][result]") {
    GenerateResult result;
    REQUIRE(result.request_id == 0);
    REQUIRE(result.full_text.empty());
    REQUIRE(result.tokens_generated == 0);
    REQUIRE(result.time_ms == 0.0);
    REQUIRE(result.tokens_per_second == 0.0);
    REQUIRE(result.multimodal_token_count == 0);
    REQUIRE_FALSE(result.used_multimodal_input);
}

TEST_CASE("GenerateResult field assignment", "[request][result]") {
    GenerateResult result;
    result.request_id = 7;
    result.full_text = "Hello world";
    result.tokens_generated = 10;
    result.time_ms = 150.5;
    result.tokens_per_second = 66.4;
    result.multimodal_token_count = 384;
    result.used_multimodal_input = true;

    REQUIRE(result.request_id == 7);
    REQUIRE(result.full_text == "Hello world");
    REQUIRE(result.tokens_generated == 10);
    REQUIRE(result.time_ms == Catch::Approx(150.5));
    REQUIRE(result.tokens_per_second == Catch::Approx(66.4));
    REQUIRE(result.multimodal_token_count == 384);
    REQUIRE(result.used_multimodal_input);
}

TEST_CASE("GenerateResult tokens_per_second calculation", "[request][result]") {
    GenerateResult result;
    result.tokens_generated = 100;
    result.time_ms = 1000.0;
    // tps = tokens / (ms / 1000) = 100 / 1.0 = 100.0
    result.tokens_per_second = (result.time_ms > 0.0) ? (result.tokens_generated / (result.time_ms / 1000.0)) : 0.0;

    REQUIRE(result.tokens_per_second == Catch::Approx(100.0));
}

TEST_CASE("GenerateResult zero time yields zero tps", "[request][result]") {
    GenerateResult result;
    result.tokens_generated = 10;
    result.time_ms = 0.0;
    result.tokens_per_second = (result.time_ms > 0.0) ? (result.tokens_generated / (result.time_ms / 1000.0)) : 0.0;

    REQUIRE(result.tokens_per_second == Catch::Approx(0.0));
}

// ---------------------------------------------------------------------------
// TokenEvent
// ---------------------------------------------------------------------------

TEST_CASE("TokenEvent defaults", "[request][token_event]") {
    TokenEvent ev;
    REQUIRE(ev.request_id == 0);
    REQUIRE(ev.text.empty());
    REQUIRE(ev.token_id == 0);
}

TEST_CASE("TokenEvent field assignment", "[request][token_event]") {
    TokenEvent ev;
    ev.request_id = 3;
    ev.text = "Hello";
    ev.token_id = 12345;

    REQUIRE(ev.request_id == 3);
    REQUIRE(ev.text == "Hello");
    REQUIRE(ev.token_id == 12345);
}

TEST_CASE("TokenEvent with empty token text", "[request][token_event]") {
    TokenEvent ev;
    ev.text = "";
    REQUIRE(ev.text.empty());
}

TEST_CASE("TokenEvent with special character token", "[request][token_event]") {
    TokenEvent ev;
    ev.text = "\n";
    REQUIRE(ev.text.size() == 1);
    REQUIRE(ev.text[0] == '\n');
}

TEST_CASE("TokenEvent with unicode token", "[request][token_event]") {
    TokenEvent ev;
    ev.text = "ü";
    REQUIRE_FALSE(ev.text.empty());
}

// ---------------------------------------------------------------------------
// ErrorEvent
// ---------------------------------------------------------------------------

TEST_CASE("ErrorEvent defaults", "[request][error_event]") {
    ErrorEvent ev;
    REQUIRE(ev.request_id == 0);
    REQUIRE(ev.code == ErrorCode::InternalError);
    REQUIRE(ev.message.empty());
    REQUIRE(ev.details.empty());
}

TEST_CASE("ErrorEvent field assignment", "[request][error_event]") {
    ErrorEvent ev;
    ev.request_id = 5;
    ev.code = ErrorCode::DecodeFailed;
    ev.message = "decode failure";
    ev.details = "llama_decode returned -1";

    REQUIRE(ev.request_id == 5);
    REQUIRE(ev.code == ErrorCode::DecodeFailed);
    REQUIRE(ev.message == "decode failure");
    REQUIRE(ev.details == "llama_decode returned -1");
}

TEST_CASE("ErrorEvent with empty details", "[request][error_event]") {
    ErrorEvent ev;
    ev.code = ErrorCode::TokenizeFailed;
    ev.message = "tokenize error";

    REQUIRE(ev.details.empty());
}

// ---------------------------------------------------------------------------
// RequestCallbacks
// ---------------------------------------------------------------------------

TEST_CASE("RequestCallbacks defaults are null", "[request][callbacks]") {
    RequestCallbacks cb;
    REQUIRE_FALSE(cb.on_token);
    REQUIRE_FALSE(cb.on_complete);
    REQUIRE_FALSE(cb.on_error);
    REQUIRE_FALSE(cb.on_cancelled);
}

TEST_CASE("RequestCallbacks on_token invocation", "[request][callbacks]") {
    RequestCallbacks cb;
    TokenEvent received;
    bool called = false;

    cb.on_token = [&](const TokenEvent &ev) {
        received = ev;
        called = true;
    };

    TokenEvent sent;
    sent.request_id = 1;
    sent.text = "test";
    sent.token_id = 42;

    cb.on_token(sent);
    REQUIRE(called);
    REQUIRE(received.request_id == 1);
    REQUIRE(received.text == "test");
    REQUIRE(received.token_id == 42);
}

TEST_CASE("RequestCallbacks on_complete invocation", "[request][callbacks]") {
    RequestCallbacks cb;
    GenerateResult received;
    bool called = false;

    cb.on_complete = [&](const GenerateResult &r) {
        received = r;
        called = true;
    };

    GenerateResult sent;
    sent.request_id = 2;
    sent.full_text = "generated";
    sent.tokens_generated = 9;

    cb.on_complete(sent);
    REQUIRE(called);
    REQUIRE(received.request_id == 2);
    REQUIRE(received.full_text == "generated");
    REQUIRE(received.tokens_generated == 9);
}

TEST_CASE("RequestCallbacks on_error invocation", "[request][callbacks]") {
    RequestCallbacks cb;
    ErrorEvent received;
    bool called = false;

    cb.on_error = [&](const ErrorEvent &ev) {
        received = ev;
        called = true;
    };

    ErrorEvent sent;
    sent.request_id = 3;
    sent.code = ErrorCode::ModelLoadFailed;
    sent.message = "bad model";

    cb.on_error(sent);
    REQUIRE(called);
    REQUIRE(received.code == ErrorCode::ModelLoadFailed);
}

TEST_CASE("RequestCallbacks on_cancelled invocation", "[request][callbacks]") {
    RequestCallbacks cb;
    RequestId received = -1;
    bool called = false;

    cb.on_cancelled = [&](RequestId id) {
        received = id;
        called = true;
    };

    cb.on_cancelled(99);
    REQUIRE(called);
    REQUIRE(received == 99);
}

TEST_CASE("RequestCallbacks null callbacks are safe to skip", "[request][callbacks]") {
    // Ensure that the pattern `if (cb.on_x) cb.on_x(...)` compiles and does not crash.
    RequestCallbacks cb;

    // These should be no-ops
    if (cb.on_token) {
        TokenEvent ev;
        cb.on_token(ev);
    }
    if (cb.on_complete) {
        GenerateResult r;
        cb.on_complete(r);
    }
    if (cb.on_error) {
        ErrorEvent ev;
        cb.on_error(ev);
    }
    if (cb.on_cancelled) {
        cb.on_cancelled(0);
    }

    // Just confirming no crash
    REQUIRE(true);
}

// ---------------------------------------------------------------------------
// RequestId type
// ---------------------------------------------------------------------------

TEST_CASE("RequestId is int32_t", "[request][request_id]") {
    static_assert(std::is_same_v<RequestId, int32_t>);
    REQUIRE(sizeof(RequestId) == 4);
}
