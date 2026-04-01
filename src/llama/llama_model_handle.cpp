#include "godot_llama/llama_model_handle.hpp"

#include <llama.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <thread>

namespace godot_llama {

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

} // namespace godot_llama
