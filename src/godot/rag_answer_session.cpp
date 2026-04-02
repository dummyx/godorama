#include "rag_answer_session.hpp"

#include "rag_corpus.hpp"
#include "rag_utils.hpp"

#include "godot_llama/rag/factories.hpp"

#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {
using namespace godot_llama;

namespace {
constexpr int kMaxAnswerQueueDepth = 32;
}

RagAnswerSession::RagAnswerSession() : generation_worker_(std::make_unique<InferenceWorker>()) {}

RagAnswerSession::~RagAnswerSession() {
    close_generation();
}

void RagAnswerSession::_bind_methods() {
    ClassDB::bind_method(D_METHOD("open_generation", "config"), &RagAnswerSession::open_generation);
    ClassDB::bind_method(D_METHOD("close_generation"), &RagAnswerSession::close_generation);
    ClassDB::bind_method(D_METHOD("is_generation_open"), &RagAnswerSession::is_generation_open);
    ClassDB::bind_method(D_METHOD("answer_async", "corpus", "question", "retrieval_options", "generation_options"),
                         &RagAnswerSession::answer_async, DEFVAL(Dictionary()), DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("cancel", "request_id"), &RagAnswerSession::cancel);
    ClassDB::bind_method(D_METHOD("poll"), &RagAnswerSession::poll);

    ADD_SIGNAL(MethodInfo("token_emitted", PropertyInfo(Variant::INT, "request_id"),
                          PropertyInfo(Variant::STRING, "token_text"), PropertyInfo(Variant::INT, "token_id")));
    ADD_SIGNAL(MethodInfo("completed", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::STRING, "text"),
                          PropertyInfo(Variant::ARRAY, "citations"), PropertyInfo(Variant::DICTIONARY, "stats")));
    ADD_SIGNAL(MethodInfo("failed", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::INT, "error_code"),
                          PropertyInfo(Variant::STRING, "error_message"), PropertyInfo(Variant::STRING, "details")));
    ADD_SIGNAL(MethodInfo("cancelled", PropertyInfo(Variant::INT, "request_id")));
}

int RagAnswerSession::open_generation(const Ref<Resource> &config) {
    close_generation();
    if (config.is_null()) {
        UtilityFunctions::push_error("RagAnswerSession: config is null");
        return static_cast<int>(ErrorCode::InvalidParameter);
    }

    ModelConfig internal_config = godot_rag::to_internal_model_config(config);
    std::shared_ptr<LlamaModelHandle> model;
    godot_llama::Error err = LlamaModelHandle::load(internal_config, model);
    if (err) {
        UtilityFunctions::push_error(String("RagAnswerSession open_generation: ") + err.message.c_str());
        return static_cast<int>(err.code);
    }

    RequestCallbacks callbacks;
    callbacks.on_token = [this](const TokenEvent &event) {
        std::lock_guard lock(event_mutex_);
        token_events_.push_back({event.request_id, event.text, event.token_id});
    };
    callbacks.on_complete = [this](const GenerateResult &result) {
        PendingRequestContext context;
        {
            std::lock_guard lock(pending_mutex_);
            auto found = pending_requests_.find(result.request_id);
            if (found != pending_requests_.end()) {
                context = found->second;
                pending_requests_.erase(found);
            }
        }

        std::lock_guard lock(event_mutex_);
        complete_events_.push_back({result.request_id, result.full_text, context.citations, context.stats});
    };
    callbacks.on_error = [this](const ErrorEvent &event) {
        {
            std::lock_guard lock(pending_mutex_);
            pending_requests_.erase(event.request_id);
        }
        std::lock_guard lock(event_mutex_);
        error_events_.push_back({event.request_id, static_cast<int>(event.code), event.message, event.details});
    };
    callbacks.on_cancelled = [this](RequestId request_id) {
        {
            std::lock_guard lock(pending_mutex_);
            pending_requests_.erase(request_id);
        }
        std::lock_guard lock(event_mutex_);
        cancel_events_.push_back({request_id});
    };

    err = generation_worker_->start(model, internal_config, std::move(callbacks));
    if (err) {
        UtilityFunctions::push_error(String("RagAnswerSession open_generation: ") + err.message.c_str());
        return static_cast<int>(err.code);
    }

    {
        std::lock_guard lock(state_mutex_);
        generation_model_ = std::move(model);
        generation_config_ = internal_config;
        packer_ = rag::make_grounded_context_packer();
    }

    generation_open_.store(true, std::memory_order_release);
    start_answer_worker();
    return static_cast<int>(ErrorCode::Ok);
}

void RagAnswerSession::close_generation() {
    stop_answer_worker();
    {
        std::lock_guard lock(pending_mutex_);
        pending_requests_.clear();
    }
    {
        std::lock_guard lock(state_mutex_);
        if (generation_worker_) {
            generation_worker_->stop();
        }
        generation_model_.reset();
        packer_.reset();
    }
    generation_open_.store(false, std::memory_order_release);
}

bool RagAnswerSession::is_generation_open() const {
    return generation_open_.load(std::memory_order_acquire);
}

int RagAnswerSession::answer_async(const Ref<RagCorpus> &corpus, const String &question,
                                   const Dictionary &retrieval_options, const Dictionary &generation_options) {
    if (!is_generation_open()) {
        UtilityFunctions::push_error("RagAnswerSession: generation model is not open");
        return -1;
    }
    if (corpus.is_null() || !corpus->is_open()) {
        UtilityFunctions::push_error("RagAnswerSession: corpus is not open");
        return -1;
    }

    auto job = std::make_shared<AnswerJob>();
    job->corpus = corpus->get_engine_shared();
    const auto utf8 = question.utf8();
    job->question.assign(utf8.get_data(), static_cast<size_t>(utf8.length()));
    job->retrieval_options = godot_rag::to_internal_retrieval_options(retrieval_options);
    job->generation_options = godot_rag::to_internal_generate_options(generation_options);
    return enqueue_answer_job(std::move(job));
}

void RagAnswerSession::cancel(int request_id) {
    {
        std::lock_guard lock(queue_mutex_);
        if (active_job_ && active_job_->id == request_id) {
            active_job_->cancelled.store(true, std::memory_order_release);
        }
        for (const auto &job : jobs_) {
            if (job->id == request_id) {
                job->cancelled.store(true, std::memory_order_release);
            }
        }
    }

    if (generation_worker_) {
        generation_worker_->cancel(request_id);
    }
}

void RagAnswerSession::poll() {
    std::vector<QueuedTokenEvent> tokens;
    std::vector<QueuedCompleteEvent> completes;
    std::vector<QueuedErrorEvent> errors;
    std::vector<QueuedCancelEvent> cancels;

    {
        std::lock_guard lock(event_mutex_);
        tokens.swap(token_events_);
        completes.swap(complete_events_);
        errors.swap(error_events_);
        cancels.swap(cancel_events_);
    }

    for (const auto &event : tokens) {
        emit_signal("token_emitted", event.request_id, String(event.text.c_str()), event.token_id);
    }
    for (const auto &event : completes) {
        emit_signal("completed", event.request_id, String(event.text.c_str()),
                    godot_rag::citations_to_array(event.citations), godot_rag::to_godot_dictionary(event.stats));
    }
    for (const auto &event : errors) {
        emit_signal("failed", event.request_id, event.error_code, String(event.message.c_str()),
                    String(event.details.c_str()));
    }
    for (const auto &event : cancels) {
        emit_signal("cancelled", event.request_id);
    }
}

void RagAnswerSession::start_answer_worker() {
    stop_answer_worker();
    running_.store(true, std::memory_order_release);
    answer_thread_ = std::jthread([this](std::stop_token) { answer_worker_loop(); });
}

void RagAnswerSession::stop_answer_worker() noexcept {
    running_.store(false, std::memory_order_release);
    queue_cv_.notify_all();
    if (answer_thread_.joinable()) {
        answer_thread_.request_stop();
        answer_thread_.join();
    }

    std::lock_guard lock(queue_mutex_);
    if (active_job_) {
        active_job_->cancelled.store(true, std::memory_order_release);
    }
    for (const auto &job : jobs_) {
        job->cancelled.store(true, std::memory_order_release);
    }
    jobs_.clear();
    active_job_.reset();
}

void RagAnswerSession::answer_worker_loop() {
    while (running_.load(std::memory_order_acquire)) {
        std::shared_ptr<AnswerJob> job;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !jobs_.empty() || !running_.load(std::memory_order_acquire);
            });
            if (!running_.load(std::memory_order_acquire)) {
                break;
            }
            job = jobs_.front();
            jobs_.pop_front();
            active_job_ = job;
        }

        std::shared_ptr<LlamaModelHandle> generation_model;
        rag::ContextPacker *packer = nullptr;
        ModelConfig generation_config;
        {
            std::lock_guard lock(state_mutex_);
            generation_model = generation_model_;
            packer = packer_.get();
            generation_config = generation_config_;
        }

        if (!generation_model || !packer || !job->corpus) {
            enqueue_error(job->id,
                          godot_llama::Error::make(godot_llama::ErrorCode::NotOpen, "RagAnswerSession is not ready"));
        } else {
            auto cancelled = [job]() { return job->cancelled.load(std::memory_order_acquire); };
            std::vector<rag::RetrievalHit> hits;
            rag::RetrievalStats retrieval_stats;
            godot_llama::Error err =
                    job->corpus->retrieve(job->question, job->retrieval_options, hits, retrieval_stats, cancelled);
            if (err) {
                enqueue_error(job->id, err);
            } else {
                rag::PromptAssembly assembly;
                err = packer->assemble(job->question, job->retrieval_options, hits, generation_model,
                                       generation_config.chat_template_override, assembly);
                if (err) {
                    enqueue_error(job->id, err);
                } else if (assembly.abstained) {
                    rag::AnswerStats stats;
                    stats.retrieval = retrieval_stats;
                    stats.abstained = true;
                    stats.prompt_style = assembly.prompt_style;

                    std::lock_guard lock(event_mutex_);
                    complete_events_.push_back({job->id, "I do not have enough evidence in the local corpus to answer that.",
                                                {}, stats});
                } else {
                    rag::AnswerStats stats;
                    stats.retrieval = retrieval_stats;
                    stats.packed_chunks = static_cast<int32_t>(assembly.packed_chunks.size());
                    stats.packed_context_tokens = assembly.context_token_count;
                    stats.truncated_chunks = assembly.truncated_chunks;
                    stats.abstained = false;
                    stats.prompt_style = assembly.prompt_style;

                    {
                        std::lock_guard lock(pending_mutex_);
                        pending_requests_[job->id] = {assembly.citations, stats};
                    }

                    (void)generation_worker_->submit_with_id(job->id, assembly.prompt, job->generation_options);
                }
            }
        }

        {
            std::lock_guard lock(queue_mutex_);
            active_job_.reset();
        }
    }
}

int RagAnswerSession::enqueue_answer_job(std::shared_ptr<AnswerJob> job) {
    const int request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    job->id = request_id;

    std::lock_guard lock(queue_mutex_);
    if (static_cast<int>(jobs_.size()) >= kMaxAnswerQueueDepth) {
        UtilityFunctions::push_error("RagAnswerSession: queue is full");
        return static_cast<int>(godot_llama::ErrorCode::QueueFull);
    }
    jobs_.push_back(std::move(job));
    queue_cv_.notify_one();
    return request_id;
}

void RagAnswerSession::enqueue_error(int request_id, const godot_llama::Error &error) {
    std::lock_guard lock(event_mutex_);
    error_events_.push_back({request_id, static_cast<int>(error.code), error.message, error.context});
}

} // namespace godot
