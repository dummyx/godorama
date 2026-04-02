#include "godot_llama/llama_context_handle.hpp"

#include "godot_llama/llama_model_handle.hpp"

#include <llama.h>

#include <algorithm>
#include <thread>

namespace godot_llama {
namespace {

bool should_abort_callback(void *data) {
    const auto *flag = static_cast<const std::atomic<bool> *>(data);
    return flag != nullptr && flag->load(std::memory_order_acquire);
}

} // namespace

LlamaContextHandle::~LlamaContextHandle() {
    if (ctx_) {
        llama_set_abort_callback(ctx_, nullptr, nullptr);
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    model_.reset();
    abort_flag_ = nullptr;
}

LlamaContextHandle::LlamaContextHandle(LlamaContextHandle &&other) noexcept
        : ctx_(other.ctx_),
          model_(std::move(other.model_)),
          abort_flag_(other.abort_flag_),
          embeddings_enabled_(other.embeddings_enabled_) {
    other.ctx_ = nullptr;
    other.abort_flag_ = nullptr;
    other.embeddings_enabled_ = false;
}

LlamaContextHandle &LlamaContextHandle::operator=(LlamaContextHandle &&other) noexcept {
    if (this != &other) {
        if (ctx_) {
            llama_set_abort_callback(ctx_, nullptr, nullptr);
            llama_free(ctx_);
        }
        ctx_ = other.ctx_;
        model_ = std::move(other.model_);
        abort_flag_ = other.abort_flag_;
        embeddings_enabled_ = other.embeddings_enabled_;
        other.ctx_ = nullptr;
        other.abort_flag_ = nullptr;
        other.embeddings_enabled_ = false;
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
    out.abort_flag_ = nullptr;
    out.embeddings_enabled_ = config.embeddings_enabled;
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

Error LlamaContextHandle::encode_tokens(std::span<const int32_t> tokens) {
    if (!ctx_) {
        return Error::make(ErrorCode::NotOpen, "Context is not valid");
    }

    if (tokens.empty()) {
        return Error::make_ok();
    }

    llama_batch batch = llama_batch_get_one(const_cast<int32_t *>(tokens.data()), static_cast<int32_t>(tokens.size()));
    const int32_t result = llama_encode(ctx_, batch);
    if (result != 0) {
        return Error::make(ErrorCode::DecodeFailed, "llama_encode failed with code " + std::to_string(result));
    }

    return Error::make_ok();
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

Error LlamaContextHandle::decode_embeddings(std::span<const float> embeddings, int32_t token_count, int32_t hidden_size,
                                            std::span<const int32_t> positions, int32_t position_components,
                                            int32_t seq_id) {
    if (!ctx_) {
        return Error::make(ErrorCode::NotOpen, "Context is not valid");
    }

    if (token_count <= 0 || hidden_size <= 0 || position_components <= 0) {
        return Error::make(ErrorCode::InvalidParameter, "Embedding decode dimensions must be positive");
    }

    if (static_cast<size_t>(token_count) * static_cast<size_t>(hidden_size) != embeddings.size()) {
        return Error::make(ErrorCode::InvalidParameter, "Embedding decode input does not match token_count * hidden_size");
    }

    if (positions.size() != static_cast<size_t>(token_count) * static_cast<size_t>(position_components)) {
        return Error::make(ErrorCode::InvalidParameter,
                           "Embedding decode positions must match token_count * position_components");
    }

    int32_t result = 0;
    if (position_components == 1) {
        llama_batch batch = llama_batch_init(token_count, hidden_size, 1);
        if (!batch.embd || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            llama_batch_free(batch);
            return Error::make(ErrorCode::InternalError, "Failed to allocate llama batch for embedding decode");
        }

        batch.n_tokens = token_count;
        std::copy(embeddings.begin(), embeddings.end(), batch.embd);
        std::copy(positions.begin(), positions.end(), batch.pos);
        for (int32_t index = 0; index < token_count; ++index) {
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = static_cast<llama_seq_id>(seq_id);
            batch.logits[index] = embeddings_enabled_ ? 1 : (index == token_count - 1 ? 1 : 0);
        }

        result = llama_decode(ctx_, batch);
        llama_batch_free(batch);
    } else {
        std::vector<llama_pos> pos_storage(positions.begin(), positions.end());
        std::vector<int32_t> n_seq_id(static_cast<size_t>(token_count), 1);
        std::vector<llama_seq_id> seq_id_storage(1, static_cast<llama_seq_id>(seq_id));
        std::vector<llama_seq_id *> seq_ids(static_cast<size_t>(token_count), seq_id_storage.data());
        std::vector<int8_t> logits(static_cast<size_t>(token_count), embeddings_enabled_ ? 1 : 0);
        if (!embeddings_enabled_) {
            logits.back() = 1;
        }

        llama_batch batch = {
                /*n_tokens =*/ token_count,
                /*token =*/ nullptr,
                /*embd =*/ const_cast<float *>(embeddings.data()),
                /*pos =*/ pos_storage.data(),
                /*n_seq_id =*/ n_seq_id.data(),
                /*seq_id =*/ seq_ids.data(),
                /*logits =*/ logits.data(),
        };

        result = llama_decode(ctx_, batch);
    }
    if (result == 2) {
        return Error::make(ErrorCode::Cancelled, "llama_decode aborted");
    }
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

float *LlamaContextHandle::get_embeddings_ith(int32_t index) const noexcept {
    if (!ctx_) {
        return nullptr;
    }
    return llama_get_embeddings_ith(ctx_, index);
}

float *LlamaContextHandle::get_embeddings_seq(int32_t seq_id) const noexcept {
    if (!ctx_) {
        return nullptr;
    }
    return llama_get_embeddings_seq(ctx_, seq_id);
}

int32_t LlamaContextHandle::pooling_type() const noexcept {
    return ctx_ ? static_cast<int32_t>(llama_pooling_type(ctx_)) : LLAMA_POOLING_TYPE_NONE;
}

void LlamaContextHandle::set_abort_flag(const std::atomic<bool> *abort_flag) noexcept {
    abort_flag_ = abort_flag;
    if (ctx_) {
        llama_set_abort_callback(ctx_, abort_flag_ ? should_abort_callback : nullptr,
                                 const_cast<std::atomic<bool> *>(abort_flag_));
    }
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
