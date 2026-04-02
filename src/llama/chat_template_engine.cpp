#include "godot_llama/chat_template_engine.hpp"

#include "chat.h"

#include <exception>

namespace godot_llama {

ChatTemplateEngine::~ChatTemplateEngine() {
    reset();
}

ChatTemplateEngine::ChatTemplateEngine(ChatTemplateEngine &&other) noexcept
    : templates_(other.templates_),
      template_override_(std::move(other.template_override_)) {
    other.templates_ = nullptr;
}

ChatTemplateEngine &ChatTemplateEngine::operator=(ChatTemplateEngine &&other) noexcept {
    if (this != &other) {
        reset();
        templates_ = other.templates_;
        template_override_ = std::move(other.template_override_);
        other.templates_ = nullptr;
    }
    return *this;
}

Error ChatTemplateEngine::initialize(const llama_model *model, std::string_view template_override) {
    reset();

    if (!model && template_override.empty()) {
        return Error::make(ErrorCode::InvalidParameter, "Chat template initialization requires a model or override");
    }

    template_override_.assign(template_override.data(), template_override.size());

    try {
        auto templates = common_chat_templates_init(model, template_override_);
        templates_ = templates.release();
    } catch (const std::exception &e) {
        reset();
        return Error::make(ErrorCode::InvalidParameter, "Failed to initialize chat template", e.what());
    }

    return Error::make_ok();
}

bool ChatTemplateEngine::is_initialized() const noexcept {
    return templates_ != nullptr;
}

const std::string &ChatTemplateEngine::template_override() const noexcept {
    return template_override_;
}

Error ChatTemplateEngine::apply(const std::vector<std::pair<std::string, std::string>> &messages,
                                bool add_assistant_turn, bool disable_thinking, std::string &out_prompt) const {
    out_prompt.clear();

    if (!templates_) {
        return Error::make(ErrorCode::CapabilityUnavailable, "Chat template engine is not initialized");
    }
    if (messages.empty()) {
        return Error::make(ErrorCode::InvalidParameter, "Cannot apply a chat template without messages");
    }

    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = add_assistant_turn;
    inputs.enable_thinking = !disable_thinking;
    inputs.messages.reserve(messages.size());

    for (const auto &message : messages) {
        common_chat_msg chat_message;
        chat_message.role = message.first;
        chat_message.content = message.second;
        inputs.messages.push_back(std::move(chat_message));
    }

    try {
        out_prompt = common_chat_templates_apply(templates_, inputs).prompt;
    } catch (const std::exception &e) {
        out_prompt.clear();
        return Error::make(ErrorCode::InternalError, "Failed to apply chat template", e.what());
    }

    return Error::make_ok();
}

void ChatTemplateEngine::reset() noexcept {
    if (templates_) {
        common_chat_templates_free(templates_);
        templates_ = nullptr;
    }
    template_override_.clear();
}

} // namespace godot_llama
