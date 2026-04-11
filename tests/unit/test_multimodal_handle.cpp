#include <catch2/catch_test_macros.hpp>

#include "godot_llama/llama_multimodal_handle.hpp"
#include "godot_llama/llama_model_handle.hpp"

using namespace godot_llama;

// ---------------------------------------------------------------------------
// LlamaMultimodalHandle default construction
// ---------------------------------------------------------------------------

TEST_CASE("LlamaMultimodalHandle default is not valid", "[multimodal_handle]") {
    LlamaMultimodalHandle handle;
    REQUIRE_FALSE(handle.is_valid());
}

TEST_CASE("LlamaMultimodalHandle default has no capabilities", "[multimodal_handle]") {
    LlamaMultimodalHandle handle;
    REQUIRE_FALSE(handle.supports_vision());
    REQUIRE_FALSE(handle.supports_audio());
    REQUIRE(handle.audio_sample_rate_hz() == -1);
}

TEST_CASE("LlamaMultimodalHandle default media_marker", "[multimodal_handle]") {
    LlamaMultimodalHandle handle;
    REQUIRE(handle.media_marker() == std::string(kDefaultMediaMarker));
}

// ---------------------------------------------------------------------------
// LlamaMultimodalHandle::create failures (no model / no file)
// ---------------------------------------------------------------------------

TEST_CASE("LlamaMultimodalHandle::create fails with null model", "[multimodal_handle]") {
    LlamaMultimodalHandle out;
    auto err = LlamaMultimodalHandle::create(nullptr, MultimodalConfig{}, out);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::ModelLoadFailed);
    REQUIRE_FALSE(out.is_valid());
}

TEST_CASE("LlamaMultimodalHandle::create fails with unloaded model", "[multimodal_handle]") {
    auto model = std::make_shared<LlamaModelHandle>();
    REQUIRE_FALSE(model->is_loaded());

    LlamaMultimodalHandle out;
    auto err = LlamaMultimodalHandle::create(model, MultimodalConfig{}, out);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::ModelLoadFailed);
    REQUIRE_FALSE(out.is_valid());
}

TEST_CASE("LlamaMultimodalHandle::create fails with empty mmproj path", "[multimodal_handle]") {
    auto model = std::make_shared<LlamaModelHandle>();

    MultimodalConfig config;
    config.mmproj_path = "";

    LlamaMultimodalHandle out;
    auto err = LlamaMultimodalHandle::create(model, config, out);
    REQUIRE(err);
    // Model is not loaded, so the first check catches ModelLoadFailed
    REQUIRE(err.code == ErrorCode::ModelLoadFailed);
    REQUIRE_FALSE(out.is_valid());
}

TEST_CASE("LlamaMultimodalHandle::create fails with nonexistent mmproj path", "[multimodal_handle]") {
    auto model = std::make_shared<LlamaModelHandle>();

    MultimodalConfig config;
    config.mmproj_path = "/nonexistent/mmproj.gguf";

    LlamaMultimodalHandle out;
    auto err = LlamaMultimodalHandle::create(model, config, out);
    REQUIRE(err);
    // Model is not loaded, so the first check catches ModelLoadFailed
    REQUIRE(err.code == ErrorCode::ModelLoadFailed);
    REQUIRE_FALSE(out.is_valid());
}

// ---------------------------------------------------------------------------
// LlamaMultimodalHandle move semantics
// ---------------------------------------------------------------------------

TEST_CASE("LlamaMultimodalHandle move from default", "[multimodal_handle]") {
    LlamaMultimodalHandle a;
    LlamaMultimodalHandle b = std::move(a);
    REQUIRE_FALSE(b.is_valid());
    REQUIRE_FALSE(b.supports_vision());
    REQUIRE_FALSE(b.supports_audio());
    REQUIRE(b.audio_sample_rate_hz() == -1);
}

TEST_CASE("LlamaMultimodalHandle move assignment from default", "[multimodal_handle]") {
    LlamaMultimodalHandle a;
    LlamaMultimodalHandle b;
    b = std::move(a);
    REQUIRE_FALSE(b.is_valid());
}

// ---------------------------------------------------------------------------
// LlamaMultimodalHandle::evaluate_prompt failures without valid context
// ---------------------------------------------------------------------------

TEST_CASE("LlamaMultimodalHandle::evaluate_prompt fails when not initialized", "[multimodal_handle]") {
    LlamaMultimodalHandle handle;
    MultimodalInput input;
    input.path = "/test/image.jpg";

    MultimodalPromptEvaluation evaluation;
    auto err = handle.evaluate_prompt(nullptr, "test prompt", {&input, 1}, true, 512, true, evaluation);
    REQUIRE(err);
    // When mtmd is not compiled in, this returns CapabilityUnavailable.
    // When mtmd is compiled in, this returns CapabilityUnavailable because ctx_ is null.
    REQUIRE(err.code == ErrorCode::CapabilityUnavailable);
}

TEST_CASE("LlamaMultimodalHandle::evaluate_prompt with empty inputs span", "[multimodal_handle]") {
    LlamaMultimodalHandle handle;
    MultimodalPromptEvaluation evaluation;
    auto err = handle.evaluate_prompt(nullptr, "test", {}, true, 512, true, evaluation);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::CapabilityUnavailable);
}

// ---------------------------------------------------------------------------
// LlamaMultimodalHandle with custom media_marker in config
// ---------------------------------------------------------------------------

TEST_CASE("LlamaMultimodalHandle config with custom marker preserves it", "[multimodal_handle]") {
    // Can't create a real handle, but we can verify the config flows correctly
    // by testing MultimodalConfig directly.
    MultimodalConfig config;
    config.mmproj_path = "/models/mmproj.gguf";
    config.media_marker = "<custom_media>";

    REQUIRE(config.media_marker == "<custom_media>");
    REQUIRE(config.mmproj_path == "/models/mmproj.gguf");
}
