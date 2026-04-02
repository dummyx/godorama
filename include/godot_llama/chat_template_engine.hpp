#pragma once

#include "godot_llama/error.hpp"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct llama_model;
struct common_chat_templates;

namespace godot_llama {

class ChatTemplateEngine {
public:
    ChatTemplateEngine() noexcept = default;
    ~ChatTemplateEngine();

    ChatTemplateEngine(const ChatTemplateEngine &) = delete;
    ChatTemplateEngine &operator=(const ChatTemplateEngine &) = delete;
    ChatTemplateEngine(ChatTemplateEngine &&other) noexcept;
    ChatTemplateEngine &operator=(ChatTemplateEngine &&other) noexcept;

    [[nodiscard]] Error initialize(const llama_model *model, std::string_view template_override);
    [[nodiscard]] bool is_initialized() const noexcept;
    [[nodiscard]] const std::string &template_override() const noexcept;

    [[nodiscard]] Error apply(const std::vector<std::pair<std::string, std::string>> &messages,
                              bool add_assistant_turn, bool disable_thinking, std::string &out_prompt) const;

private:
    void reset() noexcept;

    common_chat_templates *templates_ = nullptr;
    std::string template_override_;
};

} // namespace godot_llama
