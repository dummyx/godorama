#include "godot_llama/worker.hpp"

#include "godot_llama/llama_sampler_handle.hpp"

#include <llama.h>

#include <algorithm>
#include <chrono>

namespace godot_llama {

InferenceWorker::~InferenceWorker() {
    stop();
}

Error InferenceWorker::start(std::shared_ptr<LlamaModelHandle> model, const ModelConfig &config,
                             RequestCallbacks callbacks) {
    if (running_.load(std::memory_order_acquire)) {
        return Error::make(ErrorCode::AlreadyOpen, "Worker is already running");
    }

    model_ = std::move(model);
    config_ = config;
    callbacks_ = std::move(callbacks);

    auto err = LlamaContextHandle::create(model_, config_, context_);
    if (err) {
        return err;
    }

    if (config_.multimodal.has_value()) {
        err = LlamaMultimodalHandle::create(model_, config_.multimodal.value(), multimodal_);
        if (err) {
            context_ = {};
            model_.reset();
            return err;
        }
    }

    running_.store(true, std::memory_order_release);
    thread_ = std::jthread([this](std::stop_token) { run(); });

    return Error::make_ok();
}

void InferenceWorker::stop() noexcept {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    running_.store(false, std::memory_order_release);
    cv_.notify_all();

    if (thread_.joinable()) {
        thread_.request_stop();
        thread_.join();
    }

    // Cancel all pending requests
    std::lock_guard lock(mutex_);
    if (active_request_) {
        active_request_->cancel();
    }
    for (auto &req : queue_) {
        req->cancel();
    }
    queue_.clear();
    multimodal_ = {};
    context_ = {};
    model_.reset();
}

bool InferenceWorker::is_running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

RequestId InferenceWorker::submit(std::string prompt, GenerateOptions options,
                                  bool prompt_has_special_tokens) {
    const RequestId request_id = next_id_.fetch_add(1, std::memory_order_relaxed);
    return submit_with_id(request_id, std::move(prompt), std::move(options), prompt_has_special_tokens);
}

RequestId InferenceWorker::submit_multimodal(std::string prompt, std::vector<MultimodalInput> media_inputs,
                                             GenerateOptions options,
                                             bool prompt_has_special_tokens) {
    const RequestId request_id = next_id_.fetch_add(1, std::memory_order_relaxed);
    return submit_multimodal_with_id(request_id, std::move(prompt), std::move(media_inputs), std::move(options),
                                     prompt_has_special_tokens);
}

RequestId InferenceWorker::submit_with_id(RequestId request_id, std::string prompt, GenerateOptions options,
                                          bool prompt_has_special_tokens) {
    RequestId observed = next_id_.load(std::memory_order_relaxed);
    while (request_id >= observed &&
           !next_id_.compare_exchange_weak(observed, request_id + 1, std::memory_order_relaxed)) {
    }

    auto req = std::make_shared<GenerateRequest>();
    req->id = request_id;
    req->prompt = std::move(prompt);
    req->options = std::move(options);
    req->prompt_has_special_tokens = prompt_has_special_tokens;

    {
        std::lock_guard lock(mutex_);
        queue_.push_back(req);
    }
    cv_.notify_one();

    return req->id;
}

RequestId InferenceWorker::submit_multimodal_with_id(RequestId request_id, std::string prompt,
                                                     std::vector<MultimodalInput> media_inputs,
                                                     GenerateOptions options,
                                                     bool prompt_has_special_tokens) {
    RequestId observed = next_id_.load(std::memory_order_relaxed);
    while (request_id >= observed &&
           !next_id_.compare_exchange_weak(observed, request_id + 1, std::memory_order_relaxed)) {
    }

    auto req = std::make_shared<GenerateRequest>();
    req->id = request_id;
    req->prompt = std::move(prompt);
    req->media_inputs = std::move(media_inputs);
    req->options = std::move(options);
    req->prompt_has_special_tokens = prompt_has_special_tokens;

    {
        std::lock_guard lock(mutex_);
        queue_.push_back(req);
    }
    cv_.notify_one();

    return req->id;
}

void InferenceWorker::cancel(RequestId id) noexcept {
    std::lock_guard lock(mutex_);
    if (active_request_ && active_request_->id == id) {
        active_request_->cancel();
    }
    for (auto &req : queue_) {
        if (req->id == id) {
            req->cancel();
            break;
        }
    }
}

Error InferenceWorker::apply_chat_template(const std::vector<std::pair<std::string, std::string>> &messages,
                                           bool add_assistant_turn, std::string &out_prompt,
                                           std::vector<std::string> &out_stops) const {
    if (!model_ || !model_->is_loaded()) {
        return Error::make(ErrorCode::NotOpen, "Model is not loaded");
    }

    return model_->apply_chat_template(messages, add_assistant_turn, config_.chat_template_override,
                                       config_.disable_thinking, out_prompt, out_stops);
}

std::vector<int32_t> InferenceWorker::tokenize(std::string_view text, bool add_bos, bool special) const {
    if (!model_ || !model_->is_loaded()) {
        return {};
    }
    return model_->tokenize(text, add_bos, special);
}

std::string InferenceWorker::detokenize(const int32_t *tokens, int32_t n_tokens) const {
    if (!model_ || !model_->is_loaded()) {
        return {};
    }
    return model_->detokenize(tokens, n_tokens);
}

Error InferenceWorker::embed(std::string_view text, std::vector<float> &out) {
    if (!model_ || !model_->is_loaded()) {
        return Error::make(ErrorCode::NotOpen, "Model is not loaded");
    }
    if (!context_.is_valid()) {
        return Error::make(ErrorCode::NotOpen, "Context is not valid");
    }

    auto tokens = model_->tokenize(text, true, false);
    if (tokens.empty()) {
        return Error::make(ErrorCode::TokenizeFailed, "Tokenization produced no tokens");
    }

    context_.clear_kv_cache();

    auto err = context_.decode_tokens(tokens, 0);
    if (err) {
        return err;
    }

    float *embd = context_.get_embeddings();
    if (!embd) {
        return Error::make(ErrorCode::InternalError, "Failed to get embeddings (is embeddings_enabled set?)");
    }

    int32_t n_embd = model_->n_embd();
    out.assign(embd, embd + n_embd);
    return Error::make_ok();
}

size_t InferenceWorker::lora_adapter_count() const noexcept {
    return model_ ? model_->lora_adapter_count() : 0;
}

bool InferenceWorker::has_multimodal_session() const noexcept {
    return config_.multimodal.has_value();
}

std::string InferenceWorker::multimodal_media_marker() const {
    if (!config_.multimodal.has_value()) {
        return std::string(kDefaultMediaMarker);
    }
    return config_.multimodal->media_marker.empty() ? std::string(kDefaultMediaMarker) : config_.multimodal->media_marker;
}

bool InferenceWorker::supports_image_input() const noexcept {
    return multimodal_.supports_vision();
}

bool InferenceWorker::supports_audio_input() const noexcept {
    return multimodal_.supports_audio();
}

int32_t InferenceWorker::audio_input_sample_rate_hz() const noexcept {
    return multimodal_.audio_sample_rate_hz();
}

size_t InferenceWorker::pending_request_count() const {
    std::lock_guard lock(mutex_);
    return queue_.size();
}

std::optional<PendingRequestSnapshot> InferenceWorker::pending_request_snapshot(RequestId id) const {
    std::lock_guard lock(mutex_);
    for (const auto &req : queue_) {
        if (!req || req->id != id) {
            continue;
        }

        PendingRequestSnapshot snapshot;
        snapshot.request_id = req->id;
        snapshot.prompt = req->prompt;
        snapshot.media_inputs = req->media_inputs;
        snapshot.options = req->options;
        snapshot.prompt_has_special_tokens = req->prompt_has_special_tokens;
        snapshot.cancelled = req->is_cancelled();
        return snapshot;
    }

    return std::nullopt;
}

void InferenceWorker::run() {
    while (running_.load(std::memory_order_acquire)) {
        std::shared_ptr<GenerateRequest> req;

        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || !running_.load(std::memory_order_acquire); });

            if (!running_.load(std::memory_order_acquire)) {
                break;
            }

            if (queue_.empty()) {
                continue;
            }

            req = queue_.front();
            queue_.pop_front();
            active_request_ = req;
        }

        if (req->is_cancelled()) {
            {
                std::lock_guard lock(mutex_);
                if (active_request_ == req) {
                    active_request_.reset();
                }
            }
            if (callbacks_.on_cancelled) {
                callbacks_.on_cancelled(req->id);
            }
            continue;
        }

        process_request(*req);

        std::lock_guard lock(mutex_);
        if (active_request_ == req) {
            active_request_.reset();
        }
    }
}

void InferenceWorker::process_request(GenerateRequest &req) {
    auto t_start = std::chrono::steady_clock::now();

    // Clear KV cache for fresh generation
    context_.clear_kv_cache();

    int32_t n_past = 0;
    int32_t multimodal_token_count = 0;
    const bool used_multimodal_input = !req.media_inputs.empty();
    if (!req.media_inputs.empty()) {
        if (!multimodal_.is_valid()) {
            if (callbacks_.on_error) {
                callbacks_.on_error({req.id, ErrorCode::CapabilityUnavailable,
                                     "This session is not configured for multimodal generation",
                                     "Set LlamaModelConfig.multimodal_config before calling generate_multimodal_async"});
            }
            return;
        }

        MultimodalPromptEvaluation evaluation;
        auto err = multimodal_.evaluate_prompt(context_.raw(), req.prompt, req.media_inputs, true, config_.n_batch,
                                               true, evaluation);
        if (err) {
            if (callbacks_.on_error) {
                callbacks_.on_error({req.id, err.code, err.message, err.context});
            }
            return;
        }

        n_past = evaluation.n_past;
        multimodal_token_count = evaluation.multimodal_token_count;
    } else {
        // Chat-template-formatted prompts contain special token syntax (e.g.
        // <|turn>, <turn|>) that must be parsed into their token IDs.  The
        // template engine already strips BOS when the model's vocab says
        // add_bos=true, so passing add_special=true here re-adds it correctly.
        // Raw prompts have no special token syntax, so parse_special=false.
        const bool parse_special = req.prompt_has_special_tokens;
        auto prompt_tokens = model_->tokenize(req.prompt, /* add_special */ true, parse_special);
        if (prompt_tokens.empty()) {
            if (callbacks_.on_error) {
                callbacks_.on_error({req.id, ErrorCode::TokenizeFailed, "Prompt tokenization produced no tokens", {}});
            }
            return;
        }

        // Decode prompt
        auto err = context_.decode_tokens(prompt_tokens, 0);
        if (err) {
            if (callbacks_.on_error) {
                callbacks_.on_error({req.id, err.code, err.message, err.context});
            }
            return;
        }

        n_past = static_cast<int32_t>(prompt_tokens.size());
    }

    // Set up sampler
    LlamaSamplerHandle sampler;
    sampler.init(req.options, model_->vocab());

    std::string full_text;
    int32_t tokens_generated = 0;
    const auto *v = model_->vocab();
    int32_t eos_token = llama_vocab_eos(v);
    int32_t eot_token = llama_vocab_eot(v);

    for (int32_t i = 0; i < req.options.max_tokens; ++i) {
        if (req.is_cancelled()) {
            if (callbacks_.on_cancelled) {
                callbacks_.on_cancelled(req.id);
            }
            return;
        }

        // Sample next token
        int32_t token_id = sampler.sample(context_.raw(), -1);
        sampler.accept(token_id);

        // Check for end of generation
        if (token_id == eos_token || token_id == eot_token) {
            break;
        }

        // Convert token to text
        std::string piece = model_->token_to_piece(token_id);
        full_text += piece;
        ++tokens_generated;

        // Emit token event
        if (callbacks_.on_token) {
            callbacks_.on_token({req.id, piece, token_id});
        }

        // Check stop sequences
        bool should_stop = false;
        for (const auto &stop_seq : req.options.stop) {
            if (!stop_seq.empty() && full_text.size() >= stop_seq.size()) {
                if (full_text.compare(full_text.size() - stop_seq.size(), stop_seq.size(), stop_seq) == 0) {
                    // Remove the stop sequence from the output
                    full_text.resize(full_text.size() - stop_seq.size());
                    should_stop = true;
                    break;
                }
            }
        }
        if (should_stop) {
            break;
        }

        // Decode the new token for next iteration
        auto decode_err = context_.decode_tokens({&token_id, 1}, n_past);
        if (decode_err) {
            if (callbacks_.on_error) {
                callbacks_.on_error({req.id, decode_err.code, decode_err.message, decode_err.context});
            }
            return;
        }
        ++n_past;
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double tps = (elapsed_ms > 0.0 && tokens_generated > 0)
                         ? (static_cast<double>(tokens_generated) / (elapsed_ms / 1000.0))
                         : 0.0;

    if (callbacks_.on_complete) {
        callbacks_.on_complete(
                {req.id, std::move(full_text), tokens_generated, elapsed_ms, tps, multimodal_token_count,
                 used_multimodal_input});
    }
}

} // namespace godot_llama
