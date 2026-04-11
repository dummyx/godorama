#include <catch2/catch_test_macros.hpp>

#include "godot_llama/worker.hpp"

using namespace godot_llama;

TEST_CASE("InferenceWorker submit_multimodal_with_id preserves queued request fields",
          "[worker][multimodal]") {
    InferenceWorker worker;

    GenerateOptions options;
    options.max_tokens = 32;
    options.temperature = 0.2f;
    options.stop = {"</answer>"};
    options.seed_override = 7u;

    std::vector<MultimodalInput> media_inputs;
    media_inputs.push_back({MultimodalInputType::Image, "/tmp/example.png", "cover", {}});
    media_inputs.push_back({MultimodalInputType::Audio, "", "voice", {1, 2, 3, 4}});

    const RequestId request_id =
            worker.submit_multimodal_with_id(42, "Describe <__media__> then transcribe <__media__>.",
                                             std::move(media_inputs), options, true);

    REQUIRE(request_id == 42);
    REQUIRE(worker.pending_request_count() == 1);

    const auto snapshot = worker.pending_request_snapshot(request_id);
    const std::vector<uint8_t> expected_audio_bytes = {1, 2, 3, 4};
    const std::vector<std::string> expected_stop = {"</answer>"};
    REQUIRE(snapshot.has_value());
    REQUIRE(snapshot->request_id == 42);
    REQUIRE(snapshot->prompt == "Describe <__media__> then transcribe <__media__>.");
    REQUIRE(snapshot->media_inputs.size() == 2);
    REQUIRE(snapshot->media_inputs[0].type == MultimodalInputType::Image);
    REQUIRE(snapshot->media_inputs[0].path == "/tmp/example.png");
    REQUIRE(snapshot->media_inputs[0].id == "cover");
    REQUIRE(snapshot->media_inputs[1].type == MultimodalInputType::Audio);
    REQUIRE(snapshot->media_inputs[1].path.empty());
    REQUIRE(snapshot->media_inputs[1].id == "voice");
    REQUIRE(snapshot->media_inputs[1].data == expected_audio_bytes);
    REQUIRE(snapshot->options.max_tokens == 32);
    REQUIRE(snapshot->options.temperature == 0.2f);
    REQUIRE(snapshot->options.seed_override == 7u);
    REQUIRE(snapshot->options.stop == expected_stop);
    REQUIRE(snapshot->prompt_has_special_tokens);

    const RequestId next_text_request = worker.submit("plain text", GenerateOptions{});
    REQUIRE(next_text_request == 43);
}

TEST_CASE("InferenceWorker submit_multimodal uses monotonic request ids", "[worker][multimodal]") {
    InferenceWorker worker;

    const RequestId first =
            worker.submit_multimodal("Look at <__media__>.", {{MultimodalInputType::Image, "/tmp/a.png", "", {}}},
                                     GenerateOptions{});
    const RequestId second =
            worker.submit_multimodal("Hear <__media__>.", {{MultimodalInputType::Audio, "/tmp/a.wav", "", {}}},
                                     GenerateOptions{});
    const RequestId third = worker.submit("plain text", GenerateOptions{});

    REQUIRE(first == 1);
    REQUIRE(second == 2);
    REQUIRE(third == 3);
    REQUIRE(worker.pending_request_count() == 3);
}

TEST_CASE("InferenceWorker cancel marks queued multimodal requests", "[worker][multimodal]") {
    InferenceWorker worker;

    const RequestId request_id =
            worker.submit_multimodal("Inspect <__media__>.", {{MultimodalInputType::Image, "/tmp/a.png", "", {}}},
                                     GenerateOptions{});
    REQUIRE(worker.pending_request_count() == 1);

    worker.cancel(request_id);

    const auto snapshot = worker.pending_request_snapshot(request_id);
    REQUIRE(snapshot.has_value());
    REQUIRE(snapshot->cancelled);
}
