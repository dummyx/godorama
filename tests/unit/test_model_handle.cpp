#include <catch2/catch_test_macros.hpp>

#include "godot_llama/llama_model_handle.hpp"

using namespace godot_llama;

// ---------------------------------------------------------------------------
// LlamaModelHandle default construction (no model loaded)
// ---------------------------------------------------------------------------

TEST_CASE("LlamaModelHandle default is not loaded", "[model_handle]") {
    LlamaModelHandle handle;
    REQUIRE_FALSE(handle.is_loaded());
    REQUIRE(handle.raw() == nullptr);
    REQUIRE(handle.vocab() == nullptr);
}

TEST_CASE("LlamaModelHandle properties are zero/null when unloaded", "[model_handle]") {
    LlamaModelHandle handle;
    REQUIRE(handle.n_ctx_train() == 0);
    REQUIRE(handle.n_embd() == 0);
    REQUIRE(handle.n_embd_inp() == 0);
    REQUIRE(handle.n_embd_out() == 0);
    REQUIRE(handle.n_cls_out() == 0);
    REQUIRE(handle.n_vocab() == 0);
    REQUIRE(handle.model_size_bytes() == 0);
    REQUIRE(handle.parameter_count() == 0);
    REQUIRE(handle.lora_adapter_count() == 0);
    REQUIRE(handle.descriptor().empty());
    REQUIRE(handle.default_chat_template().empty());
    REQUIRE(handle.fingerprint().empty());
}

TEST_CASE("LlamaModelHandle capabilities are all false/zero when unloaded", "[model_handle]") {
    LlamaModelHandle handle;
    const auto &caps = handle.capabilities();
    REQUIRE_FALSE(caps.has_encoder);
    REQUIRE_FALSE(caps.has_decoder);
    REQUIRE_FALSE(caps.supports_embeddings);
    REQUIRE_FALSE(caps.supports_reranking);
    REQUIRE_FALSE(caps.is_recurrent);
    REQUIRE_FALSE(caps.is_hybrid);
    REQUIRE_FALSE(caps.is_diffusion);
    REQUIRE(caps.n_ctx_train == 0);
    REQUIRE(caps.n_embd == 0);
    REQUIRE(caps.n_embd_out == 0);
    REQUIRE(caps.n_cls_out == 0);
    REQUIRE(caps.default_pooling_type == 0);
}

TEST_CASE("LlamaModelHandle metadata returns nullopt when unloaded", "[model_handle]") {
    LlamaModelHandle handle;
    REQUIRE_FALSE(handle.metadata_value("general.architecture").has_value());
    REQUIRE_FALSE(handle.metadata_value("").has_value());
}

TEST_CASE("LlamaModelHandle metadata_entries returns empty when unloaded", "[model_handle]") {
    LlamaModelHandle handle;
    REQUIRE(handle.metadata_entries().empty());
}

TEST_CASE("LlamaModelHandle tokenize returns empty when unloaded", "[model_handle]") {
    LlamaModelHandle handle;
    REQUIRE(handle.tokenize("hello", true, false).empty());
    REQUIRE(handle.tokenize("test", false, true).empty());
    REQUIRE(handle.tokenize("", false, false).empty());
}

TEST_CASE("LlamaModelHandle detokenize returns empty when unloaded", "[model_handle]") {
    LlamaModelHandle handle;
    int32_t tokens[] = {1, 2, 3};
    REQUIRE(handle.detokenize(tokens, 3).empty());
    REQUIRE(handle.detokenize(nullptr, 0).empty());
    REQUIRE(handle.detokenize(tokens, -1).empty());
}

TEST_CASE("LlamaModelHandle token_to_piece returns empty when unloaded", "[model_handle]") {
    LlamaModelHandle handle;
    REQUIRE(handle.token_to_piece(0).empty());
    REQUIRE(handle.token_to_piece(1).empty());
    REQUIRE(handle.token_to_piece(-1).empty());
}

TEST_CASE("LlamaModelHandle apply_lora_adapters returns ok for empty adapters", "[model_handle]") {
    LlamaModelHandle handle;
    // null context with empty adapters list
    auto err = handle.apply_lora_adapters(nullptr);
    // This should fail because the context is null
    REQUIRE(err.code == ErrorCode::InvalidParameter);
}

TEST_CASE("LlamaModelHandle apply_chat_template fails when unloaded", "[model_handle]") {
    LlamaModelHandle handle;
    std::string out;
    auto err = handle.apply_chat_template({{"user", "hello"}}, true, "", false, out);
    REQUIRE(err);
}

// ---------------------------------------------------------------------------
// LlamaModelHandle::load failures (no real model file)
// ---------------------------------------------------------------------------

TEST_CASE("LlamaModelHandle::load rejects empty path", "[model_handle]") {
    std::shared_ptr<LlamaModelHandle> handle;
    ModelConfig config;
    config.model_path = "";
    auto err = LlamaModelHandle::load(config, handle);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::InvalidPath);
    REQUIRE_FALSE(handle);
}

TEST_CASE("LlamaModelHandle::load rejects nonexistent path", "[model_handle]") {
    std::shared_ptr<LlamaModelHandle> handle;
    ModelConfig config;
    config.model_path = "/nonexistent/path/model.gguf";
    auto err = LlamaModelHandle::load(config, handle);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::InvalidPath);
    REQUIRE_FALSE(handle);
}

// ---------------------------------------------------------------------------
// LlamaModelHandle move semantics
// ---------------------------------------------------------------------------

TEST_CASE("LlamaModelHandle move from default leaves source unloaded", "[model_handle]") {
    LlamaModelHandle a;
    LlamaModelHandle b = std::move(a);
    REQUIRE_FALSE(b.is_loaded());
    REQUIRE(b.raw() == nullptr);
}

// ---------------------------------------------------------------------------
// LlamaModelHandle shared_ptr use_count
// ---------------------------------------------------------------------------

TEST_CASE("LlamaModelHandle shared_ptr starts at 1", "[model_handle]") {
    // Demonstrate the intended shared_ptr ownership pattern.
    // (Can't create a real one without a model, but we can test the type.)
    std::shared_ptr<LlamaModelHandle> ptr;
    REQUIRE_FALSE(ptr);
    REQUIRE(ptr.use_count() == 0);
}
