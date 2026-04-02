#include "godot_llama/rag/corpus.hpp"

#include "godot_llama/utf8.hpp"

#include <fstream>
#include <iterator>
#include <sstream>

namespace godot_llama::rag {
namespace {

std::string normalize_newlines(std::string_view text) {
    std::string normalized;
    normalized.reserve(text.size());

    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == '\r') {
            if (i + 1 < text.size() && text[i + 1] == '\n') {
                ++i;
            }
            normalized.push_back('\n');
            continue;
        }

        normalized.push_back(ch);
    }

    return normalized;
}

std::string basename_from_path(std::string_view path) {
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string_view::npos) {
        return std::string(path);
    }
    return std::string(path.substr(pos + 1));
}

} // namespace

CorpusEngine::CorpusEngine() = default;

CorpusEngine::~CorpusEngine() {
    close();
}

Error CorpusEngine::open(const CorpusConfig &config, std::unique_ptr<CorpusStore> store, std::unique_ptr<Chunker> chunker,
                         std::unique_ptr<Embedder> embedder, std::unique_ptr<Retriever> retriever,
                         std::unique_ptr<Reranker> reranker) {
    std::lock_guard lock(mutex_);
    if (open_) {
        return Error::make(ErrorCode::AlreadyOpen, "Corpus is already open");
    }
    if (!store || !chunker || !embedder || !retriever) {
        return Error::make(ErrorCode::InvalidParameter, "Corpus dependencies are incomplete");
    }
    if (!embedder->is_open()) {
        return Error::make(ErrorCode::NotOpen, "Embedder is not open");
    }

    config_ = config;
    store_ = std::move(store);
    chunker_ = std::move(chunker);
    embedder_ = std::move(embedder);
    retriever_ = std::move(retriever);
    reranker_ = std::move(reranker);

    EmbeddingInfo embedding_state;
    embedding_state.model_fingerprint = embedder_->info().model_fingerprint;
    embedding_state.dimensions = embedder_->info().dimensions;
    embedding_state.normalized = embedder_->info().normalize_output;
    embedding_state.metric = embedder_->info().metric;
    embedding_state.pooling_type = embedder_->info().pooling_type;

    bool present = false;
    Error err = store_->get_embedding_state(embedding_state, present);
    if (err) {
        close();
        return err;
    }

    if (!present) {
        err = store_->set_embedding_state(
                {embedder_->info().model_fingerprint, embedder_->info().dimensions, embedder_->info().normalize_output,
                 embedder_->info().metric, embedder_->info().pooling_type});
        if (err) {
            close();
            return err;
        }
    }

    open_ = true;
    return Error::make_ok();
}

void CorpusEngine::close() noexcept {
    std::lock_guard lock(mutex_);
    open_ = false;
    if (store_) {
        store_->close();
    }
    store_.reset();
    chunker_.reset();
    embedder_.reset();
    retriever_.reset();
    reranker_.reset();
}

bool CorpusEngine::is_open() const noexcept {
    std::lock_guard lock(mutex_);
    return open_;
}

const CorpusConfig &CorpusEngine::config() const noexcept {
    return config_;
}

Error CorpusEngine::upsert_text(std::string source_id, std::string text, Metadata metadata, IngestStats &out_stats,
                                const ProgressCallback &on_progress, const CancelCheck &is_cancelled) {
    std::lock_guard lock(mutex_);
    const Error status = ensure_open();
    if (status) {
        return status;
    }

    NormalizedDocument document;
    Error err = normalize_document(std::move(source_id), {}, {}, std::move(text), std::move(metadata),
                                   config_.parser_mode == ParserMode::Auto ? ParserMode::Text : config_.parser_mode,
                                   document);
    if (err) {
        return err;
    }
    return ingest_document(document, out_stats, on_progress, is_cancelled);
}

Error CorpusEngine::upsert_file(const std::filesystem::path &path, Metadata metadata, IngestStats &out_stats,
                                const ProgressCallback &on_progress, const CancelCheck &is_cancelled) {
    std::lock_guard lock(mutex_);
    const Error status = ensure_open();
    if (status) {
        return status;
    }

    const auto extension = path.extension().string();
    const auto supported = std::find(config_.supported_extensions.begin(), config_.supported_extensions.end(), extension);
    if (supported == config_.supported_extensions.end()) {
        return Error::make(ErrorCode::UnsupportedFormat, "Unsupported file extension: " + extension);
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return Error::make(ErrorCode::InvalidPath, "Failed to open file: " + path.string());
    }

    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (text.find('\0') != std::string::npos) {
        return Error::make(ErrorCode::UnsupportedFormat, "Binary files are not supported for ingestion",
                           path.string());
    }

    ParserMode mode = config_.parser_mode;
    if (mode == ParserMode::Auto) {
        mode = extension == ".md" || extension == ".markdown" ? ParserMode::Markdown : ParserMode::Text;
    }

    NormalizedDocument document;
    Error err = normalize_document(path.lexically_normal().string(), basename_from_path(path.string()), path.string(),
                                   std::move(text), std::move(metadata), mode, document);
    if (err) {
        return err;
    }
    return ingest_document(document, out_stats, on_progress, is_cancelled);
}

Error CorpusEngine::delete_source(std::string_view source_id, IngestStats &out_stats) {
    std::lock_guard lock(mutex_);
    const Error status = ensure_open();
    if (status) {
        return status;
    }
    return store_->delete_source(source_id, out_stats);
}

Error CorpusEngine::clear(IngestStats &out_stats) {
    std::lock_guard lock(mutex_);
    const Error status = ensure_open();
    if (status) {
        return status;
    }
    return store_->clear(out_stats);
}

Error CorpusEngine::rebuild(IngestStats &out_stats, const ProgressCallback &on_progress, const CancelCheck &is_cancelled) {
    std::lock_guard lock(mutex_);
    const Error status = ensure_open();
    if (status) {
        return status;
    }

    std::vector<SourceRecord> sources;
    Error err = store_->list_sources(sources);
    if (err) {
        return err;
    }

    IngestStats aggregate{};
    const int32_t total = static_cast<int32_t>(sources.size());
    int32_t done = 0;
    for (const auto &source : sources) {
        if (is_cancelled && is_cancelled()) {
            return Error::make(ErrorCode::Cancelled, "Rebuild cancelled");
        }

        NormalizedDocument document;
        document.source_id = source.source_id;
        document.source_version = source.source_version;
        document.title = source.title;
        document.source_path = source.source_path;
        document.normalized_text = source.normalized_text;
        document.metadata = source.metadata;
        document.char_count = utf8::codepoint_count(source.normalized_text);

        IngestStats current;
        err = ingest_document(document, current, {}, is_cancelled);
        if (err) {
            return err;
        }

        aggregate.chunks_written += current.chunks_written;
        aggregate.chunks_reused += current.chunks_reused;
        aggregate.chunks_deleted += current.chunks_deleted;
        aggregate.embeddings_generated += current.embeddings_generated;
        aggregate.source_id = current.source_id;
        aggregate.source_version = current.source_version;

        ++done;
        if (on_progress) {
            on_progress(done, total);
        }
    }

    out_stats = aggregate;
    return Error::make_ok();
}

Error CorpusEngine::retrieve(std::string_view query, const RetrievalOptions &options, std::vector<RetrievalHit> &out_hits,
                             RetrievalStats &out_stats, const CancelCheck &is_cancelled) {
    std::lock_guard lock(mutex_);
    const Error status = ensure_open();
    if (status) {
        return status;
    }

    const Error consistency = ensure_embedding_fingerprint_consistency(true);
    if (consistency) {
        return consistency;
    }

    return retriever_->retrieve(query, options, *store_, *embedder_, reranker_.get(), out_hits, out_stats, is_cancelled);
}

Error CorpusEngine::get_stats(CorpusStats &out_stats) const {
    std::lock_guard lock(mutex_);
    const Error status = ensure_open();
    if (status) {
        return status;
    }
    return store_->get_stats(out_stats);
}

Error CorpusEngine::ensure_open() const {
    if (!open_ || !store_ || !chunker_ || !embedder_ || !retriever_) {
        return Error::make(ErrorCode::NotOpen, "Corpus is not open");
    }
    return Error::make_ok();
}

Error CorpusEngine::ensure_embedding_fingerprint_consistency(bool allow_empty_corpus) const {
    EmbeddingInfo state;
    bool present = false;
    Error err = store_->get_embedding_state(state, present);
    if (err || !present) {
        return err;
    }

    CorpusStats stats;
    err = store_->get_stats(stats);
    if (err) {
        return err;
    }
    if (allow_empty_corpus && stats.chunk_count == 0) {
        return Error::make_ok();
    }

    if (state.model_fingerprint != embedder_->info().model_fingerprint || state.dimensions != embedder_->info().dimensions ||
        state.metric != embedder_->info().metric || state.normalized != embedder_->info().normalize_output) {
        return Error::make(ErrorCode::StaleEmbeddings,
                           "Stored corpus embeddings do not match the configured embedding model",
                           state.model_fingerprint + " != " + embedder_->info().model_fingerprint);
    }

    return Error::make_ok();
}

Error CorpusEngine::normalize_document(std::string source_id, std::string title, std::string source_path, std::string text,
                                       Metadata metadata, ParserMode parser_mode, NormalizedDocument &out_document) const {
    if (source_id.empty()) {
        return Error::make(ErrorCode::InvalidParameter, "source_id is empty");
    }
    if (!utf8::is_valid(text)) {
        return Error::make(ErrorCode::InvalidUtf8, "Input text is not valid UTF-8", source_id);
    }

    out_document = {};
    out_document.source_id = std::move(source_id);
    out_document.title = title.empty() ? out_document.source_id : std::move(title);
    out_document.source_path = std::move(source_path);
    out_document.normalized_text = normalize_newlines(text);
    out_document.metadata = canonicalize_metadata(std::move(metadata));
    out_document.parser_mode = parser_mode;
    out_document.char_count = utf8::codepoint_count(out_document.normalized_text);
    out_document.source_version = make_source_version(out_document.normalized_text);
    return Error::make_ok();
}

Error CorpusEngine::ingest_document(const NormalizedDocument &document, IngestStats &out_stats,
                                    const ProgressCallback &on_progress, const CancelCheck &is_cancelled) {
    const Error consistency = ensure_embedding_fingerprint_consistency(true);
    if (consistency) {
        return consistency;
    }

    CorpusStats corpus_stats;
    Error err = store_->get_stats(corpus_stats);
    if (err) {
        return err;
    }
    if (corpus_stats.chunk_count == 0) {
        EmbeddingInfo embedding_info;
        embedding_info.model_fingerprint = embedder_->info().model_fingerprint;
        embedding_info.dimensions = embedder_->info().dimensions;
        embedding_info.normalized = embedder_->info().normalize_output;
        embedding_info.metric = embedder_->info().metric;
        embedding_info.pooling_type = embedder_->info().pooling_type;
        err = store_->set_embedding_state(embedding_info);
        if (err) {
            return err;
        }
    }

    std::optional<SourceRecord> existing;
    err = store_->get_source(document.source_id, existing);
    if (err) {
        return err;
    }
    if (existing && existing->source_version == document.source_version) {
        out_stats.source_id = document.source_id;
        out_stats.source_version = document.source_version;
        return Error::make_ok();
    }

    std::vector<ChunkRecord> chunks;
    err = chunker_->chunk(document, *embedder_, config_.chunking, chunks);
    if (err) {
        return err;
    }

    if (is_cancelled && is_cancelled()) {
        return Error::make(ErrorCode::Cancelled, "Ingestion cancelled");
    }

    std::vector<std::string> texts;
    texts.reserve(chunks.size());
    for (const auto &chunk : chunks) {
        texts.push_back(chunk.normalized_text);
    }

    std::vector<std::vector<float>> embeddings;
    err = embedder_->embed(texts, embeddings, is_cancelled);
    if (err) {
        return err;
    }
    if (embeddings.size() != chunks.size()) {
        return Error::make(ErrorCode::EmbeddingsUnavailable, "Embedder returned an unexpected vector count");
    }

    for (size_t index = 0; index < chunks.size(); ++index) {
        chunks[index].embedding = std::move(embeddings[index]);
        chunks[index].embedding_info.model_fingerprint = embedder_->info().model_fingerprint;
        chunks[index].embedding_info.dimensions = embedder_->info().dimensions;
        chunks[index].embedding_info.normalized = embedder_->info().normalize_output;
        chunks[index].embedding_info.metric = embedder_->info().metric;
        chunks[index].embedding_info.pooling_type = embedder_->info().pooling_type;

        if (on_progress) {
            on_progress(static_cast<int32_t>(index + 1), static_cast<int32_t>(chunks.size()));
        }
    }

    SourceRecord source;
    source.source_id = document.source_id;
    source.source_version = document.source_version;
    source.title = document.title;
    source.source_path = document.source_path;
    source.normalized_text = document.normalized_text;
    source.metadata = document.metadata;
    source.created_at = existing ? existing->created_at : utc_timestamp_now();
    source.updated_at = utc_timestamp_now();

    return store_->upsert_document(source, chunks, out_stats);
}

} // namespace godot_llama::rag
