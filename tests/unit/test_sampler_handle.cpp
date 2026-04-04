#include <catch2/catch_test_macros.hpp>

#include "godot_llama/llama_sampler_handle.hpp"
#include "godot_llama/llama_params.hpp"

using namespace godot_llama;

// ---------------------------------------------------------------------------
// LlamaSamplerHandle default construction
// ---------------------------------------------------------------------------

TEST_CASE("LlamaSamplerHandle default is not valid", "[sampler_handle]") {
    LlamaSamplerHandle sampler;
    REQUIRE_FALSE(sampler.is_valid());
}

TEST_CASE("LlamaSamplerHandle reset on default is safe", "[sampler_handle]") {
    LlamaSamplerHandle sampler;
    REQUIRE_NOTHROW(sampler.reset());
    REQUIRE_FALSE(sampler.is_valid());
}

// ---------------------------------------------------------------------------
// LlamaSamplerHandle move semantics
// ---------------------------------------------------------------------------

TEST_CASE("LlamaSamplerHandle move from default", "[sampler_handle]") {
    LlamaSamplerHandle a;
    LlamaSamplerHandle b = std::move(a);
    REQUIRE_FALSE(b.is_valid());
}

TEST_CASE("LlamaSamplerHandle move assignment from default", "[sampler_handle]") {
    LlamaSamplerHandle a;
    LlamaSamplerHandle b;
    b = std::move(a);
    REQUIRE_FALSE(b.is_valid());
}

// ---------------------------------------------------------------------------
// LlamaSamplerHandle init requires a vocab (not available without a model)
// ---------------------------------------------------------------------------

TEST_CASE("LlamaSamplerHandle init with null vocab is safe", "[sampler_handle]") {
    LlamaSamplerHandle sampler;
    GenerateOptions opts;
    // init with nullptr vocab should not crash.
    // The sampler might still be "valid" internally but unusable for sampling.
    REQUIRE_NOTHROW(sampler.init(opts, nullptr));
    // Whether it becomes valid depends on llama.cpp internals.
    // We just verify no crash.
}
