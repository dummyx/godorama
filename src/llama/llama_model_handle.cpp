#include "godot_llama/llama_model_handle.hpp"

#include <llama.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <thread>

namespace godot_llama {
namespace {

std::string fnv1a_hex(std::string_view value) {
    constexpr uint64_t offset = 14695981039346656037ull;
    constexpr uint64_t prime = 1099511628211ull;

    uint64_t hash = offset;
    for (const unsigned char ch : value) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= prime;
    }

    std::array<char, 17> buf{};
    snprintf(buf.data(), buf.size(), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(buf.data());
}

std::string llama_string_from_callback(const std::function<int32_t(char *, size_t)> &cb) {
    char stack_buf[512];
    int32_t len = cb(stack_buf, sizeof(stack_buf));
    if (len < 0) {
        return {};
    }
    if (static_cast<size_t>(len) < sizeof(stack_buf)) {
        return {stack_buf, static_cast<size_t>(len)};
    }

    std::vector<char> buffer(static_cast<size_t>(len) + 1, '\0');
    cb(buffer.data(), buffer.size());
    return {buffer.data(), static_cast<size_t>(len)};
}

} // namespace

LlamaModelHandle::~LlamaModelHandle() {
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

LlamaModelHandle::LlamaModelHandle(LlamaModelHandle &&other) noexcept : model_(other.model_) {
    other.model_ = nullptr;
}

LlamaModelHandle &LlamaModelHandle::operator=(LlamaModelHandle &&other) noexcept {
    if (this != &other) {
        if (model_) {
            llama_model_free(model_);
        }
        model_ = other.model_;
        other.model_ = nullptr;
    }
    return *this;
}

Error LlamaModelHandle::load(const ModelConfig &config, std::shared_ptr<LlamaModelHandle> &out) {
    namespace fs = std::filesystem;

    if (config.model_path.empty()) {
        return Error::make(ErrorCode::InvalidPath, "Model path is empty");
    }

    std::error_code ec;
    if (!fs::exists(config.model_path, ec)) {
        return Error::make(ErrorCode::InvalidPath, "Model file does not exist: " + config.model_path);
    }

    auto params = llama_model_default_params();
    params.n_gpu_layers = config.n_gpu_layers;
    params.use_mmap = config.use_mmap;
    params.use_mlock = config.use_mlock;

    llama_model *raw = llama_model_load_from_file(config.model_path.c_str(), params);
    if (!raw) {
        return Error::make(ErrorCode::ModelLoadFailed, "Failed to load model from: " + config.model_path);
    }

    auto handle = std::make_shared<LlamaModelHandle>();
    handle->model_ = raw;
    handle->refresh_metadata_cache();
    out = std::move(handle);
    return Error::make_ok();
}

bool LlamaModelHandle::is_loaded() const noexcept {
    return model_ != nullptr;
}

llama_model *LlamaModelHandle::raw() const noexcept {
    return model_;
}

const llama_vocab *LlamaModelHandle::vocab() const noexcept {
    if (!model_) {
        return nullptr;
    }
    return llama_model_get_vocab(model_);
}

int32_t LlamaModelHandle::n_ctx_train() const noexcept {
    return model_ ? llama_model_n_ctx_train(model_) : 0;
}

int32_t LlamaModelHandle::n_embd() const noexcept {
    return model_ ? llama_model_n_embd(model_) : 0;
}

int32_t LlamaModelHandle::n_embd_out() const noexcept {
    return model_ ? llama_model_n_embd_out(model_) : 0;
}

int32_t LlamaModelHandle::n_cls_out() const noexcept {
    return model_ ? static_cast<int32_t>(llama_model_n_cls_out(model_)) : 0;
}

const ModelCapabilities &LlamaModelHandle::capabilities() const noexcept {
    return capabilities_;
}

const std::string &LlamaModelHandle::descriptor() const noexcept {
    return descriptor_;
}

const std::string &LlamaModelHandle::default_chat_template() const noexcept {
    return default_chat_template_;
}

const std::string &LlamaModelHandle::fingerprint() const noexcept {
    return fingerprint_;
}

uint64_t LlamaModelHandle::model_size_bytes() const noexcept {
    return model_ ? llama_model_size(model_) : 0;
}

uint64_t LlamaModelHandle::parameter_count() const noexcept {
    return model_ ? llama_model_n_params(model_) : 0;
}

std::optional<std::string> LlamaModelHandle::metadata_value(std::string_view key) const {
    if (!model_ || key.empty()) {
        return std::nullopt;
    }

    std::string name(key);
    const std::string value = llama_string_from_callback(
            [this, &name](char *buf, size_t size) { return llama_model_meta_val_str(model_, name.c_str(), buf, size); });
    if (value.empty()) {
        return std::nullopt;
    }
    return value;
}

std::vector<std::pair<std::string, std::string>> LlamaModelHandle::metadata_entries() const {
    std::vector<std::pair<std::string, std::string>> entries;
    if (!model_) {
        return entries;
    }

    const int32_t count = llama_model_meta_count(model_);
    entries.reserve(static_cast<size_t>(std::max(count, 0)));

    for (int32_t index = 0; index < count; ++index) {
        const std::string key = llama_string_from_callback([this, index](char *buf, size_t size) {
            return llama_model_meta_key_by_index(model_, index, buf, size);
        });
        const std::string value = llama_string_from_callback([this, index](char *buf, size_t size) {
            return llama_model_meta_val_str_by_index(model_, index, buf, size);
        });
        if (!key.empty()) {
            entries.emplace_back(key, value);
        }
    }

    return entries;
}

Error LlamaModelHandle::apply_chat_template(const std::vector<std::pair<std::string, std::string>> &messages,
                                            bool add_assistant_turn, std::string_view template_override,
                                            bool disable_thinking,
                                            std::string &out_prompt) const {
    out_prompt.clear();
    if (messages.empty()) {
        return Error::make(ErrorCode::InvalidParameter, "Cannot apply a chat template without messages");
    }

    const char *template_ptr = nullptr;
    std::string template_storage;
    if (!template_override.empty()) {
        template_storage.assign(template_override.data(), template_override.size());
        template_ptr = template_storage.c_str();
    } else if (!default_chat_template_.empty()) {
        template_ptr = default_chat_template_.c_str();
    }

    if (!template_ptr) {
        return Error::make(ErrorCode::CapabilityUnavailable, "Model does not expose a chat template");
    }

    if (disable_thinking) {
        template_storage = std::string("{% set enable_thinking = false %}\n") + template_ptr;
        template_ptr = template_storage.c_str();
    }

    std::vector<llama_chat_message> chat_messages;
    chat_messages.reserve(messages.size());
    for (const auto &message : messages) {
        chat_messages.push_back({message.first.c_str(), message.second.c_str()});
    }

    std::vector<char> buffer(1024, '\0');
    int32_t len = llama_chat_apply_template(template_ptr, chat_messages.data(), chat_messages.size(),
                                            add_assistant_turn, buffer.data(),
                                            static_cast<int32_t>(buffer.size()));
    if (len < 0) {
        return Error::make(ErrorCode::InternalError, "llama_chat_apply_template failed");
    }
    if (static_cast<size_t>(len) >= buffer.size()) {
        buffer.resize(static_cast<size_t>(len) + 1, '\0');
        len = llama_chat_apply_template(template_ptr, chat_messages.data(), chat_messages.size(), add_assistant_turn,
                                        buffer.data(), static_cast<int32_t>(buffer.size()));
        if (len < 0) {
            return Error::make(ErrorCode::InternalError, "llama_chat_apply_template failed");
        }
    }

    out_prompt.assign(buffer.data(), static_cast<size_t>(len));
    return Error::make_ok();
}

std::vector<int32_t> LlamaModelHandle::tokenize(std::string_view text, bool add_bos, bool special) const {
    if (!model_) {
        return {};
    }

    const auto *v = vocab();
    // First call to get required size
    int32_t n_tokens = llama_tokenize(v, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, add_bos, special);

    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }

    std::vector<int32_t> tokens(static_cast<size_t>(n_tokens));
    int32_t actual = llama_tokenize(v, text.data(), static_cast<int32_t>(text.size()), tokens.data(), n_tokens, add_bos,
                                    special);

    if (actual < 0) {
        // Retry with larger buffer
        tokens.resize(static_cast<size_t>(-actual));
        actual = llama_tokenize(v, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                                static_cast<int32_t>(tokens.size()), add_bos, special);
    }

    if (actual >= 0) {
        tokens.resize(static_cast<size_t>(actual));
    } else {
        tokens.clear();
    }

    return tokens;
}

std::string LlamaModelHandle::detokenize(const int32_t *tokens, int32_t n_tokens) const {
    if (!model_ || !tokens || n_tokens <= 0) {
        return {};
    }

    const auto *v = vocab();
    // Estimate: up to 8 bytes per token should be plenty for most models
    std::string result;
    result.reserve(static_cast<size_t>(n_tokens) * 8);

    int32_t n =
            llama_detokenize(v, tokens, n_tokens, result.data(), static_cast<int32_t>(result.capacity()), false, false);

    if (n < 0) {
        result.resize(static_cast<size_t>(-n));
        n = llama_detokenize(v, tokens, n_tokens, result.data(), static_cast<int32_t>(result.size()), false, false);
    }

    if (n >= 0) {
        result.resize(static_cast<size_t>(n));
    } else {
        result.clear();
    }

    return result;
}

std::string LlamaModelHandle::token_to_piece(int32_t token) const {
    if (!model_) {
        return {};
    }

    const auto *v = vocab();
    char buf[128];
    int32_t n = llama_token_to_piece(v, token, buf, sizeof(buf), 0, false);

    if (n < 0) {
        std::string result(static_cast<size_t>(-n), '\0');
        llama_token_to_piece(v, token, result.data(), static_cast<int32_t>(result.size()), 0, false);
        return result;
    }

    return {buf, static_cast<size_t>(n)};
}

void LlamaModelHandle::refresh_metadata_cache() {
    capabilities_ = {};
    descriptor_.clear();
    default_chat_template_.clear();
    fingerprint_.clear();

    if (!model_) {
        return;
    }

    capabilities_.has_encoder = llama_model_has_encoder(model_);
    capabilities_.has_decoder = llama_model_has_decoder(model_);
    capabilities_.is_recurrent = llama_model_is_recurrent(model_);
    capabilities_.is_hybrid = llama_model_is_hybrid(model_);
    capabilities_.is_diffusion = llama_model_is_diffusion(model_);
    capabilities_.n_ctx_train = llama_model_n_ctx_train(model_);
    capabilities_.n_embd = llama_model_n_embd(model_);
    capabilities_.n_embd_out = llama_model_n_embd_out(model_);
    capabilities_.n_cls_out = static_cast<int32_t>(llama_model_n_cls_out(model_));

    if (const char *tmpl = llama_model_chat_template(model_, nullptr)) {
        default_chat_template_ = tmpl;
    }

    std::optional<std::string> pooling_value = metadata_value("pooling_type");
    if (!pooling_value) {
        for (const auto &[key, value] : metadata_entries()) {
            if (key.size() >= 13 && key.compare(key.size() - 13, 13, ".pooling_type") == 0) {
                pooling_value = value;
                break;
            }
        }
    }

    if (pooling_value) {
        if (*pooling_value == "mean") {
            capabilities_.default_pooling_type = LLAMA_POOLING_TYPE_MEAN;
        } else if (*pooling_value == "cls") {
            capabilities_.default_pooling_type = LLAMA_POOLING_TYPE_CLS;
        } else if (*pooling_value == "last") {
            capabilities_.default_pooling_type = LLAMA_POOLING_TYPE_LAST;
        } else if (*pooling_value == "rank") {
            capabilities_.default_pooling_type = LLAMA_POOLING_TYPE_RANK;
        }
    }

    capabilities_.supports_embeddings = capabilities_.n_embd_out > 0 ||
                                        capabilities_.default_pooling_type != LLAMA_POOLING_TYPE_NONE ||
                                        capabilities_.has_encoder;
    capabilities_.supports_reranking =
            capabilities_.default_pooling_type == LLAMA_POOLING_TYPE_RANK || capabilities_.n_cls_out > 0;

    descriptor_ = llama_string_from_callback(
            [this](char *buf, size_t size) { return llama_model_desc(model_, buf, size); });

    std::string digest_input = descriptor_;
    digest_input.append("|");
    digest_input.append(std::to_string(model_size_bytes()));
    digest_input.append("|");
    digest_input.append(std::to_string(parameter_count()));
    digest_input.append("|");
    for (const auto &[key, value] : metadata_entries()) {
        digest_input.append(key);
        digest_input.append("=");
        digest_input.append(value);
        digest_input.push_back('\n');
    }
    fingerprint_ = fnv1a_hex(digest_input);
}

} // namespace godot_llama
