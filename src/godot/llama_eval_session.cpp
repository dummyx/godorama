#include "llama_eval_session.hpp"

#include "godot_llama/llama_position_layout.hpp"
#include "llama_model_config.hpp"

#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <algorithm>

namespace godot {
namespace {

PackedFloat32Array copy_float_slice(const float *source, int32_t count) {
    PackedFloat32Array values;
    if (!source || count <= 0) {
        return values;
    }

    values.resize(count);
    for (int32_t index = 0; index < count; ++index) {
        values[index] = source[index];
    }
    return values;
}

PackedInt64Array shape_2d(int64_t rows, int64_t cols) {
    PackedInt64Array shape;
    shape.resize(2);
    shape[0] = rows;
    shape[1] = cols;
    return shape;
}

} // namespace

LlamaEvalSession::~LlamaEvalSession() {
    close();
}

void LlamaEvalSession::_bind_methods() {
    ClassDB::bind_method(D_METHOD("open", "config"), &LlamaEvalSession::open);
    ClassDB::bind_method(D_METHOD("close"), &LlamaEvalSession::close);
    ClassDB::bind_method(D_METHOD("is_open"), &LlamaEvalSession::is_open);
    ClassDB::bind_method(D_METHOD("is_opening"), &LlamaEvalSession::is_opening);
    ClassDB::bind_method(D_METHOD("run_prefill_async", "inputs_embeds", "sequence_length", "position_ids",
                                  "position_components", "logit_start", "logit_count", "include_hidden_state",
                                  "clear_kv_cache"),
                         &LlamaEvalSession::run_prefill_async, DEFVAL(PackedInt32Array()), DEFVAL(1), DEFVAL(0),
                         DEFVAL(0), DEFVAL(true), DEFVAL(true));
    ClassDB::bind_method(D_METHOD("cancel", "request_id"), &LlamaEvalSession::cancel);
    ClassDB::bind_method(D_METHOD("poll"), &LlamaEvalSession::poll);

    ADD_SIGNAL(MethodInfo("opened"));
    ADD_SIGNAL(MethodInfo("completed", PropertyInfo(Variant::INT, "request_id"),
                          PropertyInfo(Variant::DICTIONARY, "result")));
    ADD_SIGNAL(MethodInfo("failed", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::INT, "error_code"),
                          PropertyInfo(Variant::STRING, "error_message"), PropertyInfo(Variant::STRING, "details")));
    ADD_SIGNAL(MethodInfo("cancelled", PropertyInfo(Variant::INT, "request_id")));
}

int LlamaEvalSession::open(const Ref<Resource> &config) {
    if (is_open_.load(std::memory_order_acquire) || is_opening_.load(std::memory_order_acquire) ||
        !open_thread_finished_.load(std::memory_order_acquire)) {
        UtilityFunctions::push_error("LlamaEvalSession: already open, call close() first");
        return static_cast<int>(godot_llama::ErrorCode::AlreadyOpen);
    }

    if (config.is_null()) {
        UtilityFunctions::push_error("LlamaEvalSession: config is null");
        return static_cast<int>(godot_llama::ErrorCode::InvalidParameter);
    }

    auto internal_config = to_internal_config(config);

    is_opening_.store(true, std::memory_order_release);
    open_thread_finished_.store(false, std::memory_order_release);
    const uint64_t open_generation = open_generation_.fetch_add(1, std::memory_order_acq_rel) + 1;

    open_thread_ = std::jthread([this, internal_config, open_generation](std::stop_token stop_token) {
        std::shared_ptr<godot_llama::LlamaModelHandle> model;
        const auto load_err = godot_llama::LlamaModelHandle::load(internal_config, model);
        if (load_err) {
            if (!is_stale_open_generation(open_generation) && !stop_token.stop_requested()) {
                enqueue_error_event(0, load_err);
            }
            finalize_open_thread();
            return;
        }

        godot_llama::LlamaContextHandle context;
        const auto context_err = godot_llama::LlamaContextHandle::create(model, internal_config, context);
        if (context_err) {
            if (!is_stale_open_generation(open_generation) && !stop_token.stop_requested()) {
                enqueue_error_event(0, context_err);
            }
            finalize_open_thread();
            return;
        }

        if (is_stale_open_generation(open_generation) || stop_token.stop_requested()) {
            finalize_open_thread();
            return;
        }

        {
            std::lock_guard lock(state_mutex_);
            if (is_stale_open_generation(open_generation) || stop_token.stop_requested()) {
                finalize_open_thread();
                return;
            }

            model_ = std::move(model);
            context_ = std::move(context);
            worker_thread_ = std::jthread([this](std::stop_token worker_stop_token) { worker_loop(worker_stop_token); });
            is_open_.store(true, std::memory_order_release);
        }

        enqueue_opened_event();
        finalize_open_thread();
    });

    return static_cast<int>(godot_llama::ErrorCode::Ok);
}

void LlamaEvalSession::close() {
    open_generation_.fetch_add(1, std::memory_order_acq_rel);
    is_opening_.store(false, std::memory_order_release);

    if (open_thread_.joinable()) {
        open_thread_.request_stop();
        if (open_thread_finished_.load(std::memory_order_acquire)) {
            open_thread_.join();
        }
    }

    {
        std::lock_guard lock(queue_mutex_);
        if (active_request_) {
            active_request_->cancelled.store(true, std::memory_order_release);
        }
        request_queue_.clear();
    }
    queue_cv_.notify_all();

    if (worker_thread_.joinable()) {
        worker_thread_.request_stop();
        queue_cv_.notify_all();
        worker_thread_.join();
    }

    std::lock_guard lock(state_mutex_);
    if (!is_open_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    context_ = godot_llama::LlamaContextHandle();
    model_.reset();
}

bool LlamaEvalSession::is_open() const {
    return is_open_.load(std::memory_order_acquire);
}

bool LlamaEvalSession::is_opening() const {
    return is_opening_.load(std::memory_order_acquire);
}

int LlamaEvalSession::run_prefill_async(const PackedFloat32Array &inputs_embeds, int32_t sequence_length,
                                        const PackedInt32Array &position_ids, int32_t position_components,
                                        int32_t logit_start, int32_t logit_count, bool include_hidden_state,
                                        bool clear_kv_cache) {
    if (!is_open_) {
        UtilityFunctions::push_error("LlamaEvalSession: not open");
        return -static_cast<int>(godot_llama::ErrorCode::NotOpen);
    }

    if (inputs_embeds.is_empty()) {
        UtilityFunctions::push_error("LlamaEvalSession: inputs_embeds is empty");
        return -static_cast<int>(godot_llama::ErrorCode::InvalidParameter);
    }

    int32_t hidden_size = 0;
    {
        std::lock_guard lock(state_mutex_);
        if (!model_) {
            UtilityFunctions::push_error("LlamaEvalSession: model is not available");
            return -static_cast<int>(godot_llama::ErrorCode::NotOpen);
        }
        hidden_size = model_->n_embd_inp();
        if (hidden_size <= 0) {
            hidden_size = model_->n_embd();
        }
    }

    if (hidden_size <= 0) {
        UtilityFunctions::push_error("LlamaEvalSession: model reports an invalid embedding width");
        return -static_cast<int>(godot_llama::ErrorCode::InternalError);
    }

    const int32_t derived_sequence_length = static_cast<int32_t>(inputs_embeds.size() / hidden_size);
    if (sequence_length <= 0) {
        sequence_length = derived_sequence_length;
    }
    if (sequence_length <= 0 || sequence_length * hidden_size != inputs_embeds.size()) {
        UtilityFunctions::push_error("LlamaEvalSession: inputs_embeds does not match sequence_length * hidden_size");
        return -static_cast<int>(godot_llama::ErrorCode::InvalidParameter);
    }

    auto request = std::make_shared<EvalRequest>();
    request->request_id = next_request_id_.fetch_add(1, std::memory_order_acq_rel) + 1;
    request->sequence_length = sequence_length;
    request->position_components = position_components;
    request->logit_start = logit_start;
    request->logit_count = logit_count;
    request->include_hidden_state = include_hidden_state;
    request->clear_kv_cache = clear_kv_cache;

    request->embeddings.resize(static_cast<size_t>(inputs_embeds.size()));
    for (int32_t index = 0; index < inputs_embeds.size(); ++index) {
        request->embeddings[static_cast<size_t>(index)] = inputs_embeds[index];
    }

    request->positions.resize(static_cast<size_t>(position_ids.size()));
    for (int32_t index = 0; index < position_ids.size(); ++index) {
        request->positions[static_cast<size_t>(index)] = position_ids[index];
    }

    {
        std::lock_guard lock(queue_mutex_);
        request_queue_.push_back(request);
    }
    queue_cv_.notify_all();
    return request->request_id;
}

void LlamaEvalSession::cancel(int request_id) {
    std::vector<int> cancelled_queued_ids;
    {
        std::lock_guard lock(queue_mutex_);
        auto it = request_queue_.begin();
        while (it != request_queue_.end()) {
            const bool matches = request_id <= 0 || ((*it) && (*it)->request_id == request_id);
            if (!matches) {
                ++it;
                continue;
            }

            if (*it) {
                (*it)->cancelled.store(true, std::memory_order_release);
                cancelled_queued_ids.push_back((*it)->request_id);
            }
            it = request_queue_.erase(it);
        }

        if (active_request_ && (request_id <= 0 || active_request_->request_id == request_id)) {
            active_request_->cancelled.store(true, std::memory_order_release);
        }
    }
    queue_cv_.notify_all();

    for (int cancelled_id : cancelled_queued_ids) {
        enqueue_cancel_event(cancelled_id);
    }
}

void LlamaEvalSession::poll() {
    std::vector<QueuedOpenedEvent> opened;
    std::vector<QueuedCompleteEvent> completed;
    std::vector<QueuedErrorEvent> errors;
    std::vector<QueuedCancelEvent> cancelled;

    {
        std::lock_guard lock(event_mutex_);
        opened.swap(opened_events_);
        completed.swap(complete_events_);
        errors.swap(error_events_);
        cancelled.swap(cancel_events_);
    }

    for (size_t index = 0; index < opened.size(); ++index) {
        emit_signal("opened");
    }
    for (const auto &event : completed) {
        emit_signal("completed", event.request_id, event.result);
    }
    for (const auto &event : errors) {
        emit_signal("failed", event.request_id, event.error_code, event.message, event.details);
    }
    for (const auto &event : cancelled) {
        emit_signal("cancelled", event.request_id);
    }

    if (open_thread_.joinable() && open_thread_finished_.load(std::memory_order_acquire)) {
        open_thread_.join();
    }
}

godot_llama::ModelConfig LlamaEvalSession::to_internal_config(const Ref<Resource> &config) const {
    godot_llama::ModelConfig internal;

    auto *model_config = Object::cast_to<LlamaModelConfig>(config.ptr());
    if (!model_config) {
        return internal;
    }

    auto path_utf8 = model_config->get_model_path().utf8();
    internal.model_path = std::string(path_utf8.get_data(), static_cast<size_t>(path_utf8.length()));
    internal.n_ctx = model_config->get_n_ctx();
    internal.n_threads = model_config->get_n_threads();
    internal.n_batch = model_config->get_n_batch();
    internal.n_gpu_layers = model_config->get_n_gpu_layers();
    internal.seed = static_cast<uint32_t>(model_config->get_seed());
    internal.use_mmap = model_config->get_use_mmap();
    internal.use_mlock = model_config->get_use_mlock();
    internal.embeddings_enabled = model_config->get_embeddings_enabled();
    internal.disable_thinking = model_config->get_disable_thinking();
    internal.flash_attn_type = model_config->get_flash_attn_type();
    internal.type_k = model_config->get_type_k();
    internal.type_v = model_config->get_type_v();

    auto template_utf8 = model_config->get_chat_template_override().utf8();
    internal.chat_template_override =
            std::string(template_utf8.get_data(), static_cast<size_t>(template_utf8.length()));

    return internal;
}

void LlamaEvalSession::worker_loop(std::stop_token stop_token) {
    while (true) {
        std::shared_ptr<EvalRequest> request;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, stop_token, [this] { return !request_queue_.empty(); });
            if (stop_token.stop_requested() && request_queue_.empty()) {
                break;
            }
            if (request_queue_.empty()) {
                continue;
            }

            request = request_queue_.front();
            request_queue_.pop_front();
            active_request_ = request;
        }

        if (request->cancelled.load(std::memory_order_acquire)) {
            enqueue_cancel_event(request->request_id);
        } else {
            process_request(request);
        }

        std::lock_guard lock(queue_mutex_);
        if (active_request_ == request) {
            active_request_.reset();
        }
    }
}

void LlamaEvalSession::process_request(const std::shared_ptr<EvalRequest> &request) {
    std::shared_ptr<godot_llama::LlamaModelHandle> model_snapshot;
    {
        std::lock_guard lock(state_mutex_);
        model_snapshot = model_;
    }

    if (!model_snapshot || !context_.is_valid()) {
        enqueue_error_event(request->request_id, godot_llama::Error::make(godot_llama::ErrorCode::NotOpen,
                                                                          "LlamaEvalSession is not open"));
        return;
    }

    std::vector<int32_t> normalized_positions;
    const auto position_err = godot_llama::normalize_position_layout(
            request->positions, request->sequence_length, request->position_components, normalized_positions);
    if (position_err) {
        enqueue_error_event(request->request_id, position_err);
        return;
    }

    if (request->clear_kv_cache) {
        context_.clear_kv_cache();
    }

    context_.set_abort_flag(&request->cancelled);

    const int32_t input_hidden_size = std::max(model_snapshot->n_embd_inp(), model_snapshot->n_embd());
    const auto decode_err = context_.decode_embeddings(request->embeddings, request->sequence_length, input_hidden_size,
                                                       normalized_positions, request->position_components);

    context_.set_abort_flag(nullptr);

    if (decode_err.code == godot_llama::ErrorCode::Cancelled ||
        request->cancelled.load(std::memory_order_acquire)) {
        enqueue_cancel_event(request->request_id);
        return;
    }
    if (decode_err) {
        enqueue_error_event(request->request_id, decode_err);
        return;
    }

    const int32_t vocab_size = model_snapshot->n_vocab();
    if (vocab_size <= 0) {
        enqueue_error_event(request->request_id,
                            godot_llama::Error::make(godot_llama::ErrorCode::InternalError,
                                                     "Model reports an invalid vocabulary size"));
        return;
    }

    const int32_t safe_logit_start = std::clamp(request->logit_start, 0, vocab_size - 1);
    int32_t safe_logit_count = request->logit_count;
    if (safe_logit_count <= 0) {
        safe_logit_count = vocab_size - safe_logit_start;
    } else {
        safe_logit_count = std::min(safe_logit_count, vocab_size - safe_logit_start);
    }
    if (safe_logit_count <= 0) {
        enqueue_error_event(request->request_id,
                            godot_llama::Error::make(godot_llama::ErrorCode::InvalidParameter,
                                                     "Requested logit slice is empty"));
        return;
    }

    float *logits_ptr = context_.get_logits(-1);
    if (!logits_ptr) {
        enqueue_error_event(request->request_id,
                            godot_llama::Error::make(godot_llama::ErrorCode::InternalError,
                                                     "llama_get_logits_ith returned null"));
        return;
    }

    Dictionary result;
    result["logits"] = copy_float_slice(logits_ptr + safe_logit_start, safe_logit_count);
    result["logits_shape"] = shape_2d(1, safe_logit_count);

    if (request->include_hidden_state) {
        const int32_t hidden_size = model_snapshot->n_embd();
        float *hidden_base = context_.get_embeddings();
        float *hidden_ptr = nullptr;
        if (hidden_base && hidden_size > 0 && request->sequence_length > 0) {
            hidden_ptr = hidden_base + (static_cast<ptrdiff_t>(request->sequence_length - 1) * hidden_size);
        }
        if (!hidden_ptr) {
            hidden_ptr = context_.get_embeddings_ith(-1);
        }
        if (!hidden_ptr) {
            enqueue_error_event(request->request_id,
                                godot_llama::Error::make(godot_llama::ErrorCode::EmbeddingsUnavailable,
                                                         "llama_get_embeddings_ith returned null"));
            return;
        }

        result["hidden_states"] = copy_float_slice(hidden_ptr, hidden_size);
        result["hidden_states_shape"] = shape_2d(1, hidden_size);
    }

    enqueue_complete_event(request->request_id, result);
}

void LlamaEvalSession::enqueue_opened_event() {
    std::lock_guard lock(event_mutex_);
    opened_events_.push_back({});
}

void LlamaEvalSession::enqueue_complete_event(int request_id, const Dictionary &result) {
    std::lock_guard lock(event_mutex_);
    complete_events_.push_back({request_id, result});
}

void LlamaEvalSession::enqueue_error_event(int request_id, const godot_llama::Error &error) {
    std::lock_guard lock(event_mutex_);
    error_events_.push_back({request_id, static_cast<int>(error.code), String(error.message.c_str()),
                             String(error.context.c_str())});
}

void LlamaEvalSession::enqueue_cancel_event(int request_id) {
    std::lock_guard lock(event_mutex_);
    cancel_events_.push_back({request_id});
}

void LlamaEvalSession::finalize_open_thread() noexcept {
    is_opening_.store(false, std::memory_order_release);
    open_thread_finished_.store(true, std::memory_order_release);
}

bool LlamaEvalSession::is_stale_open_generation(uint64_t generation) const noexcept {
    return generation != open_generation_.load(std::memory_order_acquire);
}

} // namespace godot
