#pragma once

#include "godot_llama/rag/corpus.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace godot {

class RagCorpusConfig;

class RagCorpus : public RefCounted {
    GDCLASS(RagCorpus, RefCounted)

public:
    RagCorpus();
    ~RagCorpus() override;

    int open(const Ref<Resource> &config);
    void close();
    bool is_open() const;

    int upsert_text_async(const String &source_id, const String &text, const Dictionary &metadata = Dictionary());
    int upsert_file_async(const String &path, const Dictionary &metadata = Dictionary());
    int delete_source_async(const String &source_id);
    int clear_async();
    int rebuild_async();
    void cancel_job(int job_id);
    int retrieve_async(const String &query, const Dictionary &options = Dictionary());
    Dictionary get_stats() const;
    void poll();

    std::shared_ptr<godot_llama::rag::CorpusEngine> get_engine_shared() const;
    String get_chat_template_override() const;

protected:
    static void _bind_methods();

private:
    enum class JobKind {
        UpsertText,
        UpsertFile,
        DeleteSource,
        Clear,
        Rebuild,
        Retrieve,
    };

    struct Job {
        int id = 0;
        JobKind kind = JobKind::Retrieve;
        std::string source_id;
        std::string text;
        std::string path;
        std::string query;
        godot_llama::rag::Metadata metadata;
        godot_llama::rag::RetrievalOptions retrieval_options;
        std::atomic<bool> cancelled{false};
    };

    struct QueuedProgressEvent {
        int job_id = 0;
        int done = 0;
        int total = 0;
    };
    struct QueuedIngestCompleteEvent {
        int job_id = 0;
        godot_llama::rag::IngestStats stats;
    };
    struct QueuedRetrieveCompleteEvent {
        int request_id = 0;
        std::vector<godot_llama::rag::RetrievalHit> hits;
        godot_llama::rag::RetrievalStats stats;
    };
    struct QueuedErrorEvent {
        int id = 0;
        int error_code = 0;
        std::string message;
        std::string details;
    };

    void start_worker();
    void stop_worker() noexcept;
    void worker_loop();
    int enqueue_job(std::shared_ptr<Job> job);
    godot_llama::rag::CorpusConfig to_internal_config(const Ref<Resource> &config) const;
    void enqueue_error(int id, const godot_llama::Error &error);

    mutable std::mutex state_mutex_;
    std::shared_ptr<godot_llama::rag::CorpusEngine> engine_;
    godot_llama::rag::CorpusConfig config_;
    String chat_template_override_;

    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Job>> jobs_;
    std::shared_ptr<Job> active_job_;
    std::jthread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> next_job_id_{1};

    std::mutex event_mutex_;
    std::vector<QueuedProgressEvent> progress_events_;
    std::vector<QueuedIngestCompleteEvent> ingest_complete_events_;
    std::vector<QueuedRetrieveCompleteEvent> retrieve_complete_events_;
    std::vector<QueuedErrorEvent> error_events_;
};

} // namespace godot
