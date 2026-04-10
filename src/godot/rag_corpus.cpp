#include "rag_corpus.hpp"

#include "rag_corpus_config.hpp"
#include "rag_utils.hpp"

#include "godot_llama/rag/factories.hpp"

#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {
using namespace godot_llama;

RagCorpus::RagCorpus() = default;

RagCorpus::~RagCorpus() {
    close();
}

void RagCorpus::_bind_methods() {
    ClassDB::bind_method(D_METHOD("open", "config"), &RagCorpus::open);
    ClassDB::bind_method(D_METHOD("close"), &RagCorpus::close);
    ClassDB::bind_method(D_METHOD("is_open"), &RagCorpus::is_open);
    ClassDB::bind_method(D_METHOD("upsert_text_async", "source_id", "text", "metadata"), &RagCorpus::upsert_text_async,
                         DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("upsert_file_async", "path", "metadata"), &RagCorpus::upsert_file_async,
                         DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("delete_source_async", "source_id"), &RagCorpus::delete_source_async);
    ClassDB::bind_method(D_METHOD("clear_async"), &RagCorpus::clear_async);
    ClassDB::bind_method(D_METHOD("rebuild_async"), &RagCorpus::rebuild_async);
    ClassDB::bind_method(D_METHOD("cancel_job", "job_id"), &RagCorpus::cancel_job);
    ClassDB::bind_method(D_METHOD("retrieve_async", "query", "options"), &RagCorpus::retrieve_async,
                         DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("get_stats"), &RagCorpus::get_stats);
    ClassDB::bind_method(D_METHOD("poll"), &RagCorpus::poll);

    ADD_SIGNAL(MethodInfo("ingest_progress", PropertyInfo(Variant::INT, "job_id"), PropertyInfo(Variant::INT, "done"),
                          PropertyInfo(Variant::INT, "total")));
    ADD_SIGNAL(MethodInfo("ingest_completed", PropertyInfo(Variant::INT, "job_id"),
                          PropertyInfo(Variant::DICTIONARY, "stats")));
    ADD_SIGNAL(MethodInfo("retrieve_completed", PropertyInfo(Variant::INT, "request_id"),
                          PropertyInfo(Variant::ARRAY, "hits"), PropertyInfo(Variant::DICTIONARY, "stats")));
    ADD_SIGNAL(MethodInfo("failed", PropertyInfo(Variant::INT, "request_id_or_job_id"),
                          PropertyInfo(Variant::INT, "error_code"), PropertyInfo(Variant::STRING, "error_message"),
                          PropertyInfo(Variant::STRING, "details")));
}

int RagCorpus::open(const Ref<Resource> &config) {
    close();

    if (config.is_null()) {
        UtilityFunctions::push_error("RagCorpus: config is null");
        return static_cast<int>(ErrorCode::InvalidParameter);
    }

    auto internal_config = to_internal_config(config);
    auto engine = std::make_shared<rag::CorpusEngine>();

    std::unique_ptr<rag::CorpusStore> store;
    godot_llama::Error err = rag::make_libsql_corpus_store(internal_config, store);
    if (err) {
        UtilityFunctions::push_error(String("RagCorpus open: ") + err.message.c_str());
        return static_cast<int>(err.code);
    }

    std::unique_ptr<rag::Embedder> embedder;
    err = rag::make_llama_embedder(internal_config, embedder);
    if (err) {
        UtilityFunctions::push_error(String("RagCorpus open: ") + err.message.c_str());
        return static_cast<int>(err.code);
    }

    std::unique_ptr<rag::Reranker> reranker =
            rag::make_noop_reranker(internal_config.enable_reranker ? "configured_but_unavailable" : "disabled");

    err = engine->open(internal_config, std::move(store), rag::make_deterministic_chunker(), std::move(embedder),
                       rag::make_dense_retriever(), std::move(reranker));
    if (err) {
        UtilityFunctions::push_error(String("RagCorpus open: ") + err.message.c_str());
        return static_cast<int>(err.code);
    }

    {
        std::lock_guard lock(state_mutex_);
        engine_ = std::move(engine);
        config_ = internal_config;
    }

    start_worker();
    return static_cast<int>(ErrorCode::Ok);
}

void RagCorpus::close() {
    stop_worker();

    std::shared_ptr<rag::CorpusEngine> engine;
    {
        std::lock_guard lock(state_mutex_);
        engine.swap(engine_);
    }
    if (engine) {
        engine->close();
    }
}

bool RagCorpus::is_open() const {
    std::lock_guard lock(state_mutex_);
    return engine_ && engine_->is_open();
}

int RagCorpus::upsert_text_async(const String &source_id, const String &text, const Dictionary &metadata) {
    if (!is_open()) {
        UtilityFunctions::push_error("RagCorpus: not open");
        return -1;
    }

    rag::Metadata metadata_internal;
    godot_llama::Error err =
            godot_rag::dictionary_to_metadata(metadata, config_.max_metadata_entries, config_.max_metadata_key_bytes,
                                              config_.max_metadata_value_bytes, metadata_internal);
    if (err) {
        UtilityFunctions::push_error(String("RagCorpus metadata: ") + err.message.c_str());
        return static_cast<int>(err.code);
    }

    const auto source_id_utf8 = source_id.utf8();
    const auto text_utf8 = text.utf8();
    auto job = std::make_shared<Job>();
    job->kind = JobKind::UpsertText;
    job->source_id.assign(source_id_utf8.get_data(), static_cast<size_t>(source_id_utf8.length()));
    job->text.assign(text_utf8.get_data(), static_cast<size_t>(text_utf8.length()));
    job->metadata = std::move(metadata_internal);
    return enqueue_job(std::move(job));
}

int RagCorpus::upsert_file_async(const String &path, const Dictionary &metadata) {
    std::shared_ptr<rag::CorpusEngine> engine = get_engine_shared();
    if (!engine) {
        UtilityFunctions::push_error("RagCorpus: not open");
        return -1;
    }

    rag::Metadata metadata_internal;
    godot_llama::Error err =
            godot_rag::dictionary_to_metadata(metadata, config_.max_metadata_entries, config_.max_metadata_key_bytes,
                                              config_.max_metadata_value_bytes, metadata_internal);
    if (err) {
        UtilityFunctions::push_error(String("RagCorpus metadata: ") + err.message.c_str());
        return static_cast<int>(err.code);
    }

    const auto path_utf8 = path.utf8();
    auto job = std::make_shared<Job>();
    job->kind = JobKind::UpsertFile;
    job->path.assign(path_utf8.get_data(), static_cast<size_t>(path_utf8.length()));
    job->metadata = std::move(metadata_internal);
    return enqueue_job(std::move(job));
}

int RagCorpus::delete_source_async(const String &source_id) {
    if (!is_open()) {
        UtilityFunctions::push_error("RagCorpus: not open");
        return -1;
    }

    const auto source_id_utf8 = source_id.utf8();
    auto job = std::make_shared<Job>();
    job->kind = JobKind::DeleteSource;
    job->source_id.assign(source_id_utf8.get_data(), static_cast<size_t>(source_id_utf8.length()));
    return enqueue_job(std::move(job));
}

int RagCorpus::clear_async() {
    if (!is_open()) {
        UtilityFunctions::push_error("RagCorpus: not open");
        return -1;
    }

    auto job = std::make_shared<Job>();
    job->kind = JobKind::Clear;
    return enqueue_job(std::move(job));
}

int RagCorpus::rebuild_async() {
    if (!is_open()) {
        UtilityFunctions::push_error("RagCorpus: not open");
        return -1;
    }

    auto job = std::make_shared<Job>();
    job->kind = JobKind::Rebuild;
    return enqueue_job(std::move(job));
}

void RagCorpus::cancel_job(int job_id) {
    std::lock_guard lock(queue_mutex_);
    if (active_job_ && active_job_->id == job_id) {
        active_job_->cancelled.store(true, std::memory_order_release);
    }
    for (const auto &job : jobs_) {
        if (job->id == job_id) {
            job->cancelled.store(true, std::memory_order_release);
            break;
        }
    }
}

int RagCorpus::retrieve_async(const String &query, const Dictionary &options) {
    if (!is_open()) {
        UtilityFunctions::push_error("RagCorpus: not open");
        return -1;
    }

    const auto query_utf8 = query.utf8();
    auto job = std::make_shared<Job>();
    job->kind = JobKind::Retrieve;
    job->query.assign(query_utf8.get_data(), static_cast<size_t>(query_utf8.length()));
    job->retrieval_options = godot_rag::to_internal_retrieval_options(options);
    return enqueue_job(std::move(job));
}

Dictionary RagCorpus::get_stats() const {
    Dictionary dictionary;
    std::shared_ptr<rag::CorpusEngine> engine = get_engine_shared();
    if (!engine) {
        return dictionary;
    }

    rag::CorpusStats stats;
    const godot_llama::Error err = engine->get_stats(stats);
    if (err) {
        return dictionary;
    }
    return godot_rag::to_godot_dictionary(stats);
}

void RagCorpus::poll() {
    std::vector<QueuedProgressEvent> progress;
    std::vector<QueuedIngestCompleteEvent> ingest_done;
    std::vector<QueuedRetrieveCompleteEvent> retrieve_done;
    std::vector<QueuedErrorEvent> errors;

    {
        std::lock_guard lock(event_mutex_);
        progress.swap(progress_events_);
        ingest_done.swap(ingest_complete_events_);
        retrieve_done.swap(retrieve_complete_events_);
        errors.swap(error_events_);
    }

    for (const auto &event : progress) {
        emit_signal("ingest_progress", event.job_id, event.done, event.total);
    }
    for (const auto &event : ingest_done) {
        emit_signal("ingest_completed", event.job_id, godot_rag::to_godot_dictionary(event.stats));
    }
    for (const auto &event : retrieve_done) {
        emit_signal("retrieve_completed", event.request_id, godot_rag::retrieval_hits_to_array(event.hits),
                    godot_rag::to_godot_dictionary(event.stats));
    }
    for (const auto &event : errors) {
        emit_signal("failed", event.id, event.error_code, String(event.message.c_str()), String(event.details.c_str()));
    }
}

std::shared_ptr<rag::CorpusEngine> RagCorpus::get_engine_shared() const {
    std::lock_guard lock(state_mutex_);
    return engine_;
}

String RagCorpus::get_chat_template_override() const {
    return chat_template_override_;
}

void RagCorpus::start_worker() {
    stop_worker();
    running_.store(true, std::memory_order_release);
    worker_thread_ = std::jthread([this](std::stop_token) { worker_loop(); });
}

void RagCorpus::stop_worker() noexcept {
    running_.store(false, std::memory_order_release);
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.request_stop();
        worker_thread_.join();
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

void RagCorpus::worker_loop() {
    while (running_.load(std::memory_order_acquire)) {
        std::shared_ptr<Job> job;
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

        std::shared_ptr<rag::CorpusEngine> engine = get_engine_shared();
        if (!engine) {
            enqueue_error(job->id, godot_llama::Error::make(godot_llama::ErrorCode::NotOpen, "RagCorpus is not open"));
        } else {
            rag::IngestStats ingest_stats;
            rag::RetrievalStats retrieval_stats;
            std::vector<rag::RetrievalHit> hits;

            auto cancelled = [job]() { return job->cancelled.load(std::memory_order_acquire); };
            auto progress = [this, job](int32_t done, int32_t total) {
                std::lock_guard lock(event_mutex_);
                progress_events_.push_back({job->id, done, total});
            };

            godot_llama::Error err = godot_llama::Error::make_ok();
            switch (job->kind) {
            case JobKind::UpsertText:
                err = engine->upsert_text(job->source_id, job->text, job->metadata, ingest_stats, progress, cancelled);
                if (!err) {
                    std::lock_guard lock(event_mutex_);
                    ingest_complete_events_.push_back({job->id, ingest_stats});
                }
                break;
            case JobKind::UpsertFile:
                err = engine->upsert_file(job->path, job->metadata, ingest_stats, progress, cancelled);
                if (!err) {
                    std::lock_guard lock(event_mutex_);
                    ingest_complete_events_.push_back({job->id, ingest_stats});
                }
                break;
            case JobKind::DeleteSource:
                err = engine->delete_source(job->source_id, ingest_stats);
                if (!err) {
                    std::lock_guard lock(event_mutex_);
                    ingest_complete_events_.push_back({job->id, ingest_stats});
                }
                break;
            case JobKind::Clear:
                err = engine->clear(ingest_stats);
                if (!err) {
                    std::lock_guard lock(event_mutex_);
                    ingest_complete_events_.push_back({job->id, ingest_stats});
                }
                break;
            case JobKind::Rebuild:
                err = engine->rebuild(ingest_stats, progress, cancelled);
                if (!err) {
                    std::lock_guard lock(event_mutex_);
                    ingest_complete_events_.push_back({job->id, ingest_stats});
                }
                break;
            case JobKind::Retrieve:
                err = engine->retrieve(job->query, job->retrieval_options, hits, retrieval_stats, cancelled);
                if (!err) {
                    std::lock_guard lock(event_mutex_);
                    retrieve_complete_events_.push_back({job->id, std::move(hits), retrieval_stats});
                }
                break;
            }

            if (err) {
                enqueue_error(job->id, err);
            }
        }

        {
            std::lock_guard lock(queue_mutex_);
            active_job_.reset();
        }
    }
}

int RagCorpus::enqueue_job(std::shared_ptr<Job> job) {
    const int job_id = next_job_id_.fetch_add(1, std::memory_order_relaxed);
    job->id = job_id;

    std::lock_guard lock(queue_mutex_);
    if (static_cast<int>(jobs_.size()) >= config_.max_queue_depth) {
        UtilityFunctions::push_error("RagCorpus: queue is full");
        return static_cast<int>(ErrorCode::QueueFull);
    }
    jobs_.push_back(std::move(job));
    queue_cv_.notify_one();
    return job_id;
}

rag::CorpusConfig RagCorpus::to_internal_config(const Ref<Resource> &config) const {
    rag::CorpusConfig internal;
    RagCorpusConfig *rag_config = Object::cast_to<RagCorpusConfig>(config.ptr());
    if (!rag_config) {
        return internal;
    }

    internal.storage_path = String(rag_config->get_storage_path()).utf8().get_data();
    internal.chunking.chunk_size_tokens = rag_config->get_chunk_size_tokens();
    internal.chunking.chunk_overlap_tokens = rag_config->get_chunk_overlap_tokens();
    internal.normalize_embeddings = rag_config->get_normalize_embeddings();
    internal.vector_metric = rag::VectorMetric::Cosine;
    internal.max_batch_texts = rag_config->get_max_batch_texts();
    internal.embedding_model.model_path = String(rag_config->get_embedding_model_path()).utf8().get_data();
    internal.embedding_model.n_ctx = rag_config->get_embedding_n_ctx();
    internal.embedding_model.n_threads = rag_config->get_embedding_n_threads();
    internal.embedding_model.embeddings_enabled = true;
    internal.enable_reranker = rag_config->get_enable_reranker();
    internal.reranker_model.model_path = String(rag_config->get_reranker_model_path()).utf8().get_data();
    internal.parser_mode =
            rag::parse_parser_mode(String(rag_config->get_parser_mode()).utf8().get_data()).value_or(rag::ParserMode::Auto);
    internal.supported_extensions.clear();
    const PackedStringArray extensions = rag_config->get_supported_extensions();
    for (int64_t index = 0; index < extensions.size(); ++index) {
        internal.supported_extensions.push_back(String(extensions[index]).utf8().get_data());
    }
    return internal;
}

void RagCorpus::enqueue_error(int id, const godot_llama::Error &error) {
    std::lock_guard lock(event_mutex_);
    error_events_.push_back({id, static_cast<int>(error.code), error.message, error.context});
}

} // namespace godot
