#include <catch2/catch_test_macros.hpp>

#include "godot_llama/chat_template_engine.hpp"
#include "godot_llama/error.hpp"

#include <string>
#include <utility>
#include <vector>

using namespace godot_llama;

namespace {

constexpr const char *TEST_TEMPLATE = R"({%- for message in messages -%}
<|{{ message.role }}|>{{ message.content }}<|end|>
{%- endfor -%}
{%- if add_generation_prompt -%}
{%- if enable_thinking -%}<think>{%- else -%}<answer>{%- endif -%}
{%- endif -%})";

} // namespace

TEST_CASE("ChatTemplateEngine rejects empty initialization", "[chat_template]") {
    ChatTemplateEngine engine;
    const auto err = engine.initialize(nullptr, "");
    REQUIRE(err);
    CHECK(err.code == ErrorCode::InvalidParameter);
    CHECK_FALSE(engine.is_initialized());
}

TEST_CASE("ChatTemplateEngine applies enable_thinking correctly", "[chat_template]") {
    ChatTemplateEngine engine;
    const auto init_err = engine.initialize(nullptr, TEST_TEMPLATE);
    REQUIRE_FALSE(init_err);
    REQUIRE(engine.is_initialized());

    const std::vector<std::pair<std::string, std::string>> messages = {
        {"system", "You are concise."},
        {"user", "Say hello."},
    };

    std::string prompt;
    std::vector<std::string> stops;
    const auto thinking_err = engine.apply(messages, true, false, prompt, stops);
    REQUIRE_FALSE(thinking_err);
    CHECK(prompt.find("<think>") != std::string::npos);
    CHECK(prompt.find("<answer>") == std::string::npos);

    const auto non_thinking_err = engine.apply(messages, true, true, prompt, stops);
    REQUIRE_FALSE(non_thinking_err);
    CHECK(prompt.find("<answer>") != std::string::npos);
    CHECK(prompt.find("<think>") == std::string::npos);
}
