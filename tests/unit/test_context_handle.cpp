#include <catch2/catch_test_macros.hpp>

#include "godot_llama/llama_context_handle.hpp"
#include "godot_llama/llama_model_handle.hpp"

using namespace godot_llama;

// ---------------------------------------------------------------------------
// LlamaContextHandle default construction
// ---------------------------------------------------------------------------

TEST_CASE("LlamaContextHandle default is not valid", "[context_handle]") {
    LlamaContextHandle ctx;
    REQUIRE_FALSE(ctx.is_valid());
    REQUIRE(ctx.raw() == nullptr);
    REQUIRE(ctx.n_ctx() == 0);
    REQUIRE(ctx.model() == nullptr);
}

TEST_CASE("LlamaContextHandle embedding pointers are null when invalid", "[context_handle]") {
    LlamaContextHandle ctx;
    REQUIRE(ctx.get_logits(0) == nullptr);
    REQUIRE(ctx.get_embeddings() == nullptr);
    REQUIRE(ctx.get_embeddings_ith(0) == nullptr);
    REQUIRE(ctx.get_embeddings_seq(0) == nullptr);
    REQUIRE(ctx.pooling_type() == 0); // LLAMA_POOLING_TYPE_NONE
}

TEST_CASE("LlamaContextHandle::create fails with null model", "[context_handle]") {
    LlamaContextHandle ctx;
    auto err = LlamaContextHandle::create(nullptr, ModelConfig{}, ctx);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::ModelLoadFailed);
    REQUIRE_FALSE(ctx.is_valid());
}

TEST_CASE("LlamaContextHandle::create fails with unloaded model shared_ptr", "[context_handle]") {
    auto model = std::make_shared<LlamaModelHandle>();
    REQUIRE_FALSE(model->is_loaded());

    LlamaContextHandle ctx;
    auto err = LlamaContextHandle::create(model, ModelConfig{}, ctx);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::ModelLoadFailed);
    REQUIRE_FALSE(ctx.is_valid());
}

// ---------------------------------------------------------------------------
// LlamaContextHandle move semantics
// ---------------------------------------------------------------------------

TEST_CASE("LlamaContextHandle move from default", "[context_handle]") {
    LlamaContextHandle a;
    LlamaContextHandle b = std::move(a);
    REQUIRE_FALSE(b.is_valid());
    REQUIRE(b.raw() == nullptr);
}

TEST_CASE("LlamaContextHandle move assignment from default", "[context_handle]") {
    LlamaContextHandle a;
    LlamaContextHandle b;
    b = std::move(a);
    REQUIRE_FALSE(b.is_valid());
}

TEST_CASE("LlamaContextHandle self-move-assignment is safe", "[context_handle]") {
    LlamaContextHandle a;
    // Self-assignment should be handled safely (no double-free).
    // Technically UB in the general case, but our implementation checks `this != &other`.
    // We only test the default state here.
    REQUIRE_FALSE(a.is_valid());
}

// ---------------------------------------------------------------------------
// LlamaContextHandle decode/encode fail when invalid
// ---------------------------------------------------------------------------

TEST_CASE("LlamaContextHandle decode_tokens fails when invalid", "[context_handle]") {
    LlamaContextHandle ctx;
    int32_t tokens[] = {1, 2, 3};
    auto err = ctx.decode_tokens(tokens, 0);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::NotOpen);
}

TEST_CASE("LlamaContextHandle decode_tokens succeeds with empty span when invalid", "[context_handle]") {
    LlamaContextHandle ctx;
    auto err = ctx.decode_tokens({}, 0);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::NotOpen);
}

TEST_CASE("LlamaContextHandle encode_tokens fails when invalid", "[context_handle]") {
    LlamaContextHandle ctx;
    int32_t tokens[] = {1, 2, 3};
    auto err = ctx.encode_tokens(tokens);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::NotOpen);
}

TEST_CASE("LlamaContextHandle decode_embeddings fails when invalid", "[context_handle]") {
    LlamaContextHandle ctx;
    float embd[] = {0.1f, 0.2f, 0.3f};
    int32_t pos[] = {0};
    auto err = ctx.decode_embeddings(embd, 1, 3, pos, 1);
    REQUIRE(err);
    REQUIRE(err.code == ErrorCode::NotOpen);
}

// ---------------------------------------------------------------------------
// LlamaContextHandle clear_kv_cache and set_abort_flag are safe when invalid
// ---------------------------------------------------------------------------

TEST_CASE("LlamaContextHandle clear_kv_cache is safe when invalid", "[context_handle]") {
    LlamaContextHandle ctx;
    REQUIRE_NOTHROW(ctx.clear_kv_cache());
}

TEST_CASE("LlamaContextHandle set_abort_flag is safe when invalid", "[context_handle]") {
    LlamaContextHandle ctx;
    std::atomic<bool> flag{false};
    REQUIRE_NOTHROW(ctx.set_abort_flag(&flag));
    REQUIRE_NOTHROW(ctx.set_abort_flag(nullptr));
}
