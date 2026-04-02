#pragma once

#include "godot_llama/llama_model_handle.hpp"
#include "godot_llama/request.hpp"
#include "godot_llama/rag/corpus.hpp"
#include "godot_llama/rag/interfaces.hpp"
#include "godot_llama/worker.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace godot {

class RagCorpus;

class RagAnswerSession : public RefCounted {
    GDCLASS(RagAnswerSession, RefCounted)

public:
    RagAnswerSession();
    ~RagAnswerSession() override;

    int open_generation(const Ref<Resource> &config);
    void close_generation();
    bool is_generation_open() const;

    int answer_async(const Ref<RagCorpus> &corpus, const String &question, const Dictionary &retrieval_options = Dictionary(),
                     const Dictionary &generation_options = Dictionary());
    void cancel(int request_id);
    void poll();

protected:
    static void _bind_methods();

private:
    struct AnswerJob {
        int id = 0;
        std::shared_ptr<godot_llama::rag::CorpusEngine> corpus;
        std::string question;
        godot_llama::rag::RetrievalOptions retrieval_options;
        godot_llama::GenerateOptions generation_options;
        std::atomic<bool> cancelled{false};
    };

    struct PendingRequestContext {
        std::vector<godot_llama::rag::Citation> citations;
        godot_llama::rag::AnswerStats stats;
    };

    struct QueuedTokenEvent {
        int request_id = 0;
        std::string text;
        int token_id = 0;
    };
    struct QueuedCompleteEvent {
        int request_id = 0;
        std::string text;
        std::vector<godot_llama::rag::Citation> citations;
        godot_llama::rag::AnswerStats stats;
    };
    struct QueuedErrorEvent {
        int request_id = 0;
        int error_code = 0;
        std::string message;
        std::string details;
    };
    struct QueuedCancelEvent {
        int request_id = 0;
    };

    void start_answer_worker();
    void stop_answer_worker() noexcept;
    void answer_worker_loop();
    int enqueue_answer_job(std::shared_ptr<AnswerJob> job);
    void enqueue_error(int request_id, const godot_llama::Error &error);

    mutable std::mutex state_mutex_;
    std::unique_ptr<godot_llama::InferenceWorker> generation_worker_;
    std::shared_ptr<godot_llama::LlamaModelHandle> generation_model_;
    godot_llama::ModelConfig generation_config_;
    std::unique_ptr<godot_llama::rag::ContextPacker> packer_;
    std::atomic<bool> generation_open_{false};

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<AnswerJob>> jobs_;
    std::shared_ptr<AnswerJob> active_job_;
    std::jthread answer_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> next_request_id_{1};

    std::mutex pending_mutex_;
    std::unordered_map<int, PendingRequestContext> pending_requests_;

    std::mutex event_mutex_;
    std::vector<QueuedTokenEvent> token_events_;
    std::vector<QueuedCompleteEvent> complete_events_;
    std::vector<QueuedErrorEvent> error_events_;
    std::vector<QueuedCancelEvent> cancel_events_;
};

} // namespace godot
