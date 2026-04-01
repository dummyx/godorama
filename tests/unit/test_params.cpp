#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "godot_llama/llama_params.hpp"

using namespace godot_llama;

TEST_CASE("ModelConfig defaults", "[params]") {
    ModelConfig config;
    REQUIRE(config.n_ctx == 2048);
    REQUIRE(config.n_threads == -1);
    REQUIRE(config.n_batch == 512);
    REQUIRE(config.n_gpu_layers == 0);
    REQUIRE(config.seed == 0xFFFFFFFF);
    REQUIRE(config.use_mmap == true);
    REQUIRE(config.use_mlock == false);
    REQUIRE(config.embeddings_enabled == false);
    REQUIRE(config.model_path.empty());
    REQUIRE(config.chat_template_override.empty());
}

TEST_CASE("GenerateOptions defaults", "[params]") {
    GenerateOptions opts;
    REQUIRE(opts.max_tokens == 256);
    REQUIRE(opts.temperature == Catch::Approx(0.8f));
    REQUIRE(opts.top_p == Catch::Approx(0.95f));
    REQUIRE(opts.top_k == 40);
    REQUIRE(opts.min_p == Catch::Approx(0.05f));
    REQUIRE(opts.repeat_penalty == Catch::Approx(1.1f));
    REQUIRE(opts.stop.empty());
    REQUIRE_FALSE(opts.seed_override.has_value());
}

TEST_CASE("GenerateOptions seed override", "[params]") {
    GenerateOptions opts;
    opts.seed_override = 42u;
    REQUIRE(opts.seed_override.has_value());
    REQUIRE(opts.seed_override.value() == 42u);
}
