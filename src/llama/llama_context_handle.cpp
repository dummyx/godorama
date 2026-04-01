#include "godot_llama/llama_context_handle.hpp"

#include "godot_llama/llama_model_handle.hpp"

#include <llama.h>

#include <algorithm>
#include <thread>

namespace godot_llama {

LlamaContextHandle::~LlamaContextHandle() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    model_.reset();
}

LlamaContextHandle::LlamaContextHandle(LlamaContextHandle &&other) noexcept
        : ctx_(other.ctx_), model_(std::move(other.model_)) {
    other.ctx_ = nullptr;
}

LlamaContextHandle &LlamaContextHandle::operator=(LlamaContextHandle &&other) noexcept {
    if (this != &other) {
        if (ctx_) {
            llama_free(ctx_);
        }
        ctx_ = other.ctx_;
        model_ = std::move(other.model_);
        other.ctx_ = nullptr;
    }
    return *this;
}

Error LlamaContextHandle::create(const std::shared_ptr<LlamaModelHandle> &model, const ModelConfig &config,
                                 LlamaContextHandle &out) {
    if (!model || !model->is_loaded()) {
        return Error::make(ErrorCode::ModelLoadFailed, "Model is not loaded");
    }

    auto params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(config.n_ctx);
    params.n_batch = static_cast<uint32_t>(config.n_batch);
    params.n_ubatch = static_cast<uint32_t>(config.n_batch);

    if (config.n_threads > 0) {
        params.n_threads = config.n_threads;
        params.n_threads_batch = config.n_threads;
    } else {
        auto hw = static_cast<int32_t>(std::thread::hardware_concurrency());
        if (hw <= 0) {
            hw = 4;
        }
        params.n_threads = hw;
        params.n_threads_batch = hw;
    }

    params.embeddings = config.embeddings_enabled;

    llama_context *ctx = llama_init_from_model(model->raw(), params);
    if (!ctx) {
        return Error::make(ErrorCode::ContextCreateFailed, "Failed to create llama context");
    }

    out.ctx_ = ctx;
    out.model_ = model;
    return Error::make_ok();
}

bool LlamaContextHandle::is_valid() const noexcept {
    return ctx_ != nullptr;
}

llama_context *LlamaContextHandle::raw() const noexcept {
    return ctx_;
}

const std::shared_ptr<LlamaModelHandle> &LlamaContextHandle::model() const noexcept {
    return model_;
}

Error LlamaContextHandle::decode_tokens(std::span<const int32_t> tokens, int32_t pos_offset) {
    if (!ctx_) {
        return Error::make(ErrorCode::NotOpen, "Context is not valid");
    }

    if (tokens.empty()) {
        return Error::make_ok();
    }

    llama_batch batch = llama_batch_get_one(const_cast<int32_t *>(tokens.data()), static_cast<int32_t>(tokens.size()));

    int32_t result = llama_decode(ctx_, batch);
    if (result != 0) {
        return Error::make(ErrorCode::DecodeFailed, "llama_decode failed with code " + std::to_string(result));
    }

    return Error::make_ok();
}

float *LlamaContextHandle::get_logits(int32_t idx) const noexcept {
    if (!ctx_) {
        return nullptr;
    }
    return llama_get_logits_ith(ctx_, idx);
}

float *LlamaContextHandle::get_embeddings() const noexcept {
    if (!ctx_) {
        return nullptr;
    }
    return llama_get_embeddings(ctx_);
}

void LlamaContextHandle::clear_kv_cache() noexcept {
    if (ctx_) {
        llama_memory_clear(llama_get_memory(ctx_), true);
    }
}

int32_t LlamaContextHandle::n_ctx() const noexcept {
    return ctx_ ? static_cast<int32_t>(llama_n_ctx(ctx_)) : 0;
}

} // namespace godot_llama
