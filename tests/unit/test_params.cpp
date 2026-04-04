#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "godot_llama/llama_params.hpp"

#include <limits>
#include <optional>
#include <vector>

using namespace godot_llama;

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

TEST_CASE("ModelConfig defaults", "[params][model_config]") {
    ModelConfig config;
    REQUIRE(config.model_path.empty());
    REQUIRE(config.n_ctx == 2048);
    REQUIRE(config.n_threads == -1);
    REQUIRE(config.n_batch == 512);
    REQUIRE(config.n_gpu_layers == 0);
    REQUIRE(config.seed == 0xFFFFFFFF);
    REQUIRE(config.use_mmap == true);
    REQUIRE(config.use_mlock == false);
    REQUIRE(config.embeddings_enabled == false);
    REQUIRE(config.disable_thinking == false);
    REQUIRE(config.chat_template_override.empty());
    REQUIRE(config.lora_adapters.empty());
    REQUIRE_FALSE(config.multimodal.has_value());
}

TEST_CASE("ModelConfig field assignment", "[params][model_config]") {
    ModelConfig config;
    config.model_path = "/data/models/test.gguf";
    config.n_ctx = 4096;
    config.n_threads = 8;
    config.n_batch = 1024;
    config.n_gpu_layers = 12;
    config.seed = 42;
    config.use_mmap = false;
    config.use_mlock = true;
    config.embeddings_enabled = true;
    config.disable_thinking = true;
    config.chat_template_override = "chatml";

    REQUIRE(config.model_path == "/data/models/test.gguf");
    REQUIRE(config.n_ctx == 4096);
    REQUIRE(config.n_threads == 8);
    REQUIRE(config.n_batch == 1024);
    REQUIRE(config.n_gpu_layers == 12);
    REQUIRE(config.seed == 42u);
    REQUIRE(config.use_mmap == false);
    REQUIRE(config.use_mlock == true);
    REQUIRE(config.embeddings_enabled == true);
    REQUIRE(config.disable_thinking == true);
    REQUIRE(config.chat_template_override == "chatml");
}

TEST_CASE("ModelConfig with LoRA adapters", "[params][model_config]") {
    ModelConfig config;

    LoraAdapterConfig adapter1;
    adapter1.path = "/models/lora1.gguf";
    adapter1.scale = 0.8f;

    LoraAdapterConfig adapter2;
    adapter2.path = "/models/lora2.gguf";
    adapter2.scale = 1.5f;

    config.lora_adapters = {adapter1, adapter2};

    REQUIRE(config.lora_adapters.size() == 2);
    REQUIRE(config.lora_adapters[0].path == "/models/lora1.gguf");
    REQUIRE(config.lora_adapters[0].scale == Catch::Approx(0.8f));
    REQUIRE(config.lora_adapters[1].path == "/models/lora2.gguf");
    REQUIRE(config.lora_adapters[1].scale == Catch::Approx(1.5f));
}

TEST_CASE("ModelConfig with multimodal config", "[params][model_config]") {
    ModelConfig config;
    REQUIRE_FALSE(config.multimodal.has_value());

    MultimodalConfig mm;
    mm.mmproj_path = "/models/mmproj.gguf";
    mm.media_marker = "<image>";
    mm.use_gpu = true;
    mm.print_timings = true;
    mm.n_threads = 4;
    mm.image_min_tokens = 64;
    mm.image_max_tokens = 512;

    config.multimodal = mm;
    REQUIRE(config.multimodal.has_value());
    REQUIRE(config.multimodal->mmproj_path == "/models/mmproj.gguf");
    REQUIRE(config.multimodal->media_marker == "<image>");
    REQUIRE(config.multimodal->use_gpu == true);
    REQUIRE(config.multimodal->print_timings == true);
    REQUIRE(config.multimodal->n_threads == 4);
    REQUIRE(config.multimodal->image_min_tokens == 64);
    REQUIRE(config.multimodal->image_max_tokens == 512);
}

TEST_CASE("ModelConfig multimodal can be reset", "[params][model_config]") {
    ModelConfig config;
    config.multimodal = MultimodalConfig{};
    REQUIRE(config.multimodal.has_value());

    config.multimodal.reset();
    REQUIRE_FALSE(config.multimodal.has_value());
}

TEST_CASE("ModelConfig seed boundary values", "[params][model_config]") {
    ModelConfig config;

    config.seed = 0;
    REQUIRE(config.seed == 0u);

    config.seed = std::numeric_limits<uint32_t>::max();
    REQUIRE(config.seed == std::numeric_limits<uint32_t>::max());

    config.seed = 1;
    REQUIRE(config.seed == 1u);
}

TEST_CASE("ModelConfig copy preserves all fields", "[params][model_config]") {
    ModelConfig original;
    original.model_path = "/test.gguf";
    original.n_ctx = 8192;
    original.n_threads = 4;
    original.embeddings_enabled = true;
    original.lora_adapters.push_back({"/lora.gguf", 0.5f});
    original.multimodal = MultimodalConfig{"/mmproj.gguf", "<__media__>", false, false, -1, 0, 0};

    ModelConfig copy = original;

    REQUIRE(copy.model_path == "/test.gguf");
    REQUIRE(copy.n_ctx == 8192);
    REQUIRE(copy.n_threads == 4);
    REQUIRE(copy.embeddings_enabled == true);
    REQUIRE(copy.lora_adapters.size() == 1);
    REQUIRE(copy.lora_adapters[0].path == "/lora.gguf");
    REQUIRE(copy.multimodal.has_value());
    REQUIRE(copy.multimodal->mmproj_path == "/mmproj.gguf");
}

// ---------------------------------------------------------------------------
// GenerateOptions
// ---------------------------------------------------------------------------

TEST_CASE("GenerateOptions defaults", "[params][generate_options]") {
    GenerateOptions opts;
    REQUIRE(opts.max_tokens == 256);
    REQUIRE(opts.temperature == Catch::Approx(0.8f));
    REQUIRE(opts.top_p == Catch::Approx(0.95f));
    REQUIRE(opts.top_k == 40);
    REQUIRE(opts.min_p == Catch::Approx(0.05f));
    REQUIRE(opts.repeat_penalty == Catch::Approx(1.1f));
    REQUIRE(opts.repeat_last_n == 64);
    REQUIRE(opts.stop.empty());
    REQUIRE_FALSE(opts.seed_override.has_value());
}

TEST_CASE("GenerateOptions field assignment", "[params][generate_options]") {
    GenerateOptions opts;
    opts.max_tokens = 1024;
    opts.temperature = 0.0f;
    opts.top_p = 1.0f;
    opts.top_k = 100;
    opts.min_p = 0.1f;
    opts.repeat_penalty = 1.2f;
    opts.repeat_last_n = 128;
    opts.stop = {"\n", "###", "<|end|>"};
    opts.seed_override = 12345u;

    REQUIRE(opts.max_tokens == 1024);
    REQUIRE(opts.temperature == Catch::Approx(0.0f));
    REQUIRE(opts.top_p == Catch::Approx(1.0f));
    REQUIRE(opts.top_k == 100);
    REQUIRE(opts.min_p == Catch::Approx(0.1f));
    REQUIRE(opts.repeat_penalty == Catch::Approx(1.2f));
    REQUIRE(opts.repeat_last_n == 128);
    REQUIRE(opts.stop.size() == 3);
    REQUIRE(opts.stop[0] == "\n");
    REQUIRE(opts.stop[1] == "###");
    REQUIRE(opts.stop[2] == "<|end|>");
    REQUIRE(opts.seed_override.has_value());
    REQUIRE(opts.seed_override.value() == 12345u);
}

TEST_CASE("GenerateOptions seed override set/clear", "[params][generate_options]") {
    GenerateOptions opts;
    REQUIRE_FALSE(opts.seed_override.has_value());

    opts.seed_override = 42u;
    REQUIRE(opts.seed_override.has_value());
    REQUIRE(opts.seed_override.value() == 42u);

    opts.seed_override.reset();
    REQUIRE_FALSE(opts.seed_override.has_value());
}

TEST_CASE("GenerateOptions stop sequences edge cases", "[params][generate_options]") {
    GenerateOptions opts;

    SECTION("empty stop sequence list") {
        REQUIRE(opts.stop.empty());
    }

    SECTION("single-character stop") {
        opts.stop = {"."};
        REQUIRE(opts.stop.size() == 1);
    }

    SECTION("long stop sequence") {
        std::string long_stop(256, 'x');
        opts.stop = {long_stop};
        REQUIRE(opts.stop[0].size() == 256);
    }

    SECTION("many stop sequences") {
        opts.stop = {"a", "b", "c", "d", "e"};
        REQUIRE(opts.stop.size() == 5);
    }
}

TEST_CASE("GenerateOptions temperature extremes", "[params][generate_options]") {
    GenerateOptions opts;

    SECTION("greedy (temperature 0)") {
        opts.temperature = 0.0f;
        REQUIRE(opts.temperature == Catch::Approx(0.0f));
    }

    SECTION("high temperature") {
        opts.temperature = 2.0f;
        REQUIRE(opts.temperature == Catch::Approx(2.0f));
    }
}

TEST_CASE("GenerateOptions max_tokens boundary", "[params][generate_options]") {
    GenerateOptions opts;

    SECTION("single token") {
        opts.max_tokens = 1;
        REQUIRE(opts.max_tokens == 1);
    }

    SECTION("large token count") {
        opts.max_tokens = 32768;
        REQUIRE(opts.max_tokens == 32768);
    }

    SECTION("zero tokens") {
        opts.max_tokens = 0;
        REQUIRE(opts.max_tokens == 0);
    }
}

// ---------------------------------------------------------------------------
// LoraAdapterConfig
// ---------------------------------------------------------------------------

TEST_CASE("LoraAdapterConfig defaults", "[params][lora]") {
    LoraAdapterConfig lora;
    REQUIRE(lora.path.empty());
    REQUIRE(lora.scale == Catch::Approx(1.0f));
}

TEST_CASE("LoraAdapterConfig assignment", "[params][lora]") {
    LoraAdapterConfig lora;
    lora.path = "/path/to/adapter.gguf";
    lora.scale = 0.75f;

    REQUIRE(lora.path == "/path/to/adapter.gguf");
    REQUIRE(lora.scale == Catch::Approx(0.75f));
}

TEST_CASE("LoraAdapterConfig scale boundary values", "[params][lora]") {
    LoraAdapterConfig lora;

    lora.scale = 0.0f;
    REQUIRE(lora.scale == Catch::Approx(0.0f));

    lora.scale = 100.0f;
    REQUIRE(lora.scale == Catch::Approx(100.0f));
}

// ---------------------------------------------------------------------------
// MultimodalConfig
// ---------------------------------------------------------------------------

TEST_CASE("MultimodalConfig defaults", "[params][multimodal]") {
    MultimodalConfig mm;
    REQUIRE(mm.mmproj_path.empty());
    REQUIRE(mm.media_marker == std::string(kDefaultMediaMarker));
    REQUIRE(mm.use_gpu == false);
    REQUIRE(mm.print_timings == false);
    REQUIRE(mm.n_threads == -1);
    REQUIRE(mm.image_min_tokens == 0);
    REQUIRE(mm.image_max_tokens == 0);
}

TEST_CASE("MultimodalConfig field assignment", "[params][multimodal]") {
    MultimodalConfig mm;
    mm.mmproj_path = "/models/mmproj.gguf";
    mm.media_marker = "<custom_marker>";
    mm.use_gpu = true;
    mm.print_timings = true;
    mm.n_threads = 2;
    mm.image_min_tokens = 128;
    mm.image_max_tokens = 1024;

    REQUIRE(mm.mmproj_path == "/models/mmproj.gguf");
    REQUIRE(mm.media_marker == "<custom_marker>");
    REQUIRE(mm.use_gpu == true);
    REQUIRE(mm.print_timings == true);
    REQUIRE(mm.n_threads == 2);
    REQUIRE(mm.image_min_tokens == 128);
    REQUIRE(mm.image_max_tokens == 1024);
}

TEST_CASE("MultimodalConfig media_marker default constant", "[params][multimodal]") {
    REQUIRE(std::string(kDefaultMediaMarker) == "<__media__>");
    REQUIRE(kDefaultMediaMarker[0] == '<');
}

TEST_CASE("MultimodalConfig empty media_marker preserves default", "[params][multimodal]") {
    // The handle implementation uses kDefaultMediaMarker when the marker is empty.
    // Here we just confirm the config stores what was set.
    MultimodalConfig mm;
    mm.media_marker = "";
    REQUIRE(mm.media_marker.empty());

    mm.media_marker = kDefaultMediaMarker;
    REQUIRE(mm.media_marker == std::string(kDefaultMediaMarker));
}

// ---------------------------------------------------------------------------
// MultimodalInputType
// ---------------------------------------------------------------------------

TEST_CASE("MultimodalInputType enum values", "[params][multimodal]") {
    REQUIRE(MultimodalInputType::Image == MultimodalInputType::Image);
    REQUIRE(MultimodalInputType::Audio == MultimodalInputType::Audio);
    REQUIRE(MultimodalInputType::Image != MultimodalInputType::Audio);

    // Verify underlying values for serialization stability
    REQUIRE(static_cast<uint8_t>(MultimodalInputType::Image) == 0);
    REQUIRE(static_cast<uint8_t>(MultimodalInputType::Audio) == 1);
}

// ---------------------------------------------------------------------------
// MultimodalInput
// ---------------------------------------------------------------------------

TEST_CASE("MultimodalInput defaults", "[params][multimodal]") {
    MultimodalInput input;
    REQUIRE(input.type == MultimodalInputType::Image);
    REQUIRE(input.path.empty());
    REQUIRE(input.id.empty());
    REQUIRE(input.data.empty());
}

TEST_CASE("MultimodalInput file-path image", "[params][multimodal]") {
    MultimodalInput input;
    input.type = MultimodalInputType::Image;
    input.path = "/images/photo.jpg";

    REQUIRE(input.type == MultimodalInputType::Image);
    REQUIRE(input.path == "/images/photo.jpg");
    REQUIRE(input.data.empty());
}

TEST_CASE("MultimodalInput file-path audio", "[params][multimodal]") {
    MultimodalInput input;
    input.type = MultimodalInputType::Audio;
    input.path = "/audio/recording.wav";

    REQUIRE(input.type == MultimodalInputType::Audio);
    REQUIRE(input.path == "/audio/recording.wav");
    REQUIRE(input.data.empty());
}

TEST_CASE("MultimodalInput in-memory buffer", "[params][multimodal]") {
    MultimodalInput input;
    input.type = MultimodalInputType::Image;
    input.data = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46}; // JPEG header

    REQUIRE(input.data.size() == 8);
    REQUIRE(input.data[0] == 0xFF);
    REQUIRE(input.data[1] == 0xD8);
    REQUIRE(input.path.empty());
}

TEST_CASE("MultimodalInput both path and data", "[params][multimodal]") {
    // When both are present, data takes precedence at the consumption layer.
    // Here we just confirm the struct holds both.
    MultimodalInput input;
    input.path = "/images/photo.png";
    input.data = {0x89, 0x50, 0x4E, 0x47}; // PNG header

    REQUIRE_FALSE(input.path.empty());
    REQUIRE_FALSE(input.data.empty());
}

TEST_CASE("MultimodalInput with id", "[params][multimodal]") {
    MultimodalInput input;
    input.path = "/images/photo.jpg";
    input.id = "kv_slot_42";

    REQUIRE(input.id == "kv_slot_42");
}

TEST_CASE("MultimodalInput empty id is default", "[params][multimodal]") {
    MultimodalInput input;
    input.path = "/images/photo.jpg";
    REQUIRE(input.id.empty());
}

TEST_CASE("MultimodalInput large buffer", "[params][multimodal]") {
    // Simulate a realistic image buffer (e.g., 1 MB)
    MultimodalInput input;
    input.data.resize(1024 * 1024, 0xAB);
    REQUIRE(input.data.size() == 1024 * 1024);
    REQUIRE(input.data.front() == 0xAB);
    REQUIRE(input.data.back() == 0xAB);
}

TEST_CASE("MultimodalInput vector of mixed inputs", "[params][multimodal]") {
    std::vector<MultimodalInput> inputs;

    MultimodalInput img1;
    img1.type = MultimodalInputType::Image;
    img1.path = "/images/a.jpg";
    inputs.push_back(std::move(img1));

    MultimodalInput audio1;
    audio1.type = MultimodalInputType::Audio;
    audio1.path = "/audio/b.wav";
    inputs.push_back(std::move(audio1));

    MultimodalInput img2;
    img2.type = MultimodalInputType::Image;
    img2.data = {0xFF, 0xD8, 0xFF};
    inputs.push_back(std::move(img2));

    REQUIRE(inputs.size() == 3);
    REQUIRE(inputs[0].type == MultimodalInputType::Image);
    REQUIRE(inputs[0].path == "/images/a.jpg");
    REQUIRE(inputs[1].type == MultimodalInputType::Audio);
    REQUIRE(inputs[1].path == "/audio/b.wav");
    REQUIRE(inputs[2].type == MultimodalInputType::Image);
    REQUIRE(inputs[2].data.size() == 3);
    REQUIRE(inputs[2].path.empty());
}

// ---------------------------------------------------------------------------
// Struct copy/move semantics
// ---------------------------------------------------------------------------

TEST_CASE("ModelConfig is copyable", "[params][model_config]") {
    ModelConfig a;
    a.model_path = "/orig.gguf";
    a.lora_adapters.push_back({"/lora.gguf", 0.5f});
    a.multimodal = MultimodalConfig{};

    ModelConfig b = a;
    REQUIRE(b.model_path == "/orig.gguf");
    REQUIRE(b.lora_adapters.size() == 1);
    REQUIRE(b.multimodal.has_value());

    // Mutating copy does not affect original
    b.model_path = "/copy.gguf";
    REQUIRE(a.model_path == "/orig.gguf");
    REQUIRE(b.model_path == "/copy.gguf");
}

TEST_CASE("GenerateOptions is copyable", "[params][generate_options]") {
    GenerateOptions a;
    a.max_tokens = 500;
    a.seed_override = 99u;
    a.stop = {"###"};

    GenerateOptions b = a;
    REQUIRE(b.max_tokens == 500);
    REQUIRE(b.seed_override.has_value());
    REQUIRE(b.stop.size() == 1);

    b.seed_override.reset();
    REQUIRE(a.seed_override.has_value());
    REQUIRE_FALSE(b.seed_override.has_value());
}

TEST_CASE("MultimodalInput data copy is independent", "[params][multimodal]") {
    MultimodalInput a;
    a.data = {1, 2, 3, 4, 5};

    MultimodalInput b = a;
    REQUIRE(b.data == a.data);

    b.data[0] = 99;
    REQUIRE(a.data[0] == 1);
    REQUIRE(b.data[0] == 99);
}
