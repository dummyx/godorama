#include "rag_corpus_config.hpp"

#include <godot_cpp/core/class_db.hpp>

namespace godot {

RagCorpusConfig::RagCorpusConfig() {
    supported_extensions_.push_back(".txt");
    supported_extensions_.push_back(".md");
    supported_extensions_.push_back(".markdown");
}

void RagCorpusConfig::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_storage_path", "path"), &RagCorpusConfig::set_storage_path);
    ClassDB::bind_method(D_METHOD("get_storage_path"), &RagCorpusConfig::get_storage_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "storage_path", PROPERTY_HINT_SAVE_FILE, "*.sqlite3,*.db"),
                 "set_storage_path", "get_storage_path");

    ClassDB::bind_method(D_METHOD("set_chunk_size_tokens", "value"), &RagCorpusConfig::set_chunk_size_tokens);
    ClassDB::bind_method(D_METHOD("get_chunk_size_tokens"), &RagCorpusConfig::get_chunk_size_tokens);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "chunk_size_tokens"), "set_chunk_size_tokens", "get_chunk_size_tokens");

    ClassDB::bind_method(D_METHOD("set_chunk_overlap_tokens", "value"), &RagCorpusConfig::set_chunk_overlap_tokens);
    ClassDB::bind_method(D_METHOD("get_chunk_overlap_tokens"), &RagCorpusConfig::get_chunk_overlap_tokens);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "chunk_overlap_tokens"), "set_chunk_overlap_tokens",
                 "get_chunk_overlap_tokens");

    ClassDB::bind_method(D_METHOD("set_normalize_embeddings", "value"), &RagCorpusConfig::set_normalize_embeddings);
    ClassDB::bind_method(D_METHOD("get_normalize_embeddings"), &RagCorpusConfig::get_normalize_embeddings);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "normalize_embeddings"), "set_normalize_embeddings",
                 "get_normalize_embeddings");

    ClassDB::bind_method(D_METHOD("set_vector_metric", "value"), &RagCorpusConfig::set_vector_metric);
    ClassDB::bind_method(D_METHOD("get_vector_metric"), &RagCorpusConfig::get_vector_metric);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "vector_metric", PROPERTY_HINT_ENUM, "cosine,dot"), "set_vector_metric",
                 "get_vector_metric");

    ClassDB::bind_method(D_METHOD("set_max_batch_texts", "value"), &RagCorpusConfig::set_max_batch_texts);
    ClassDB::bind_method(D_METHOD("get_max_batch_texts"), &RagCorpusConfig::get_max_batch_texts);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_batch_texts"), "set_max_batch_texts", "get_max_batch_texts");

    ClassDB::bind_method(D_METHOD("set_embedding_model_path", "value"), &RagCorpusConfig::set_embedding_model_path);
    ClassDB::bind_method(D_METHOD("get_embedding_model_path"), &RagCorpusConfig::get_embedding_model_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "embedding_model_path", PROPERTY_HINT_FILE, "*.gguf"),
                 "set_embedding_model_path", "get_embedding_model_path");

    ClassDB::bind_method(D_METHOD("set_embedding_n_ctx", "value"), &RagCorpusConfig::set_embedding_n_ctx);
    ClassDB::bind_method(D_METHOD("get_embedding_n_ctx"), &RagCorpusConfig::get_embedding_n_ctx);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "embedding_n_ctx"), "set_embedding_n_ctx", "get_embedding_n_ctx");

    ClassDB::bind_method(D_METHOD("set_embedding_n_threads", "value"), &RagCorpusConfig::set_embedding_n_threads);
    ClassDB::bind_method(D_METHOD("get_embedding_n_threads"), &RagCorpusConfig::get_embedding_n_threads);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "embedding_n_threads"), "set_embedding_n_threads",
                 "get_embedding_n_threads");

    ClassDB::bind_method(D_METHOD("set_enable_reranker", "value"), &RagCorpusConfig::set_enable_reranker);
    ClassDB::bind_method(D_METHOD("get_enable_reranker"), &RagCorpusConfig::get_enable_reranker);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable_reranker"), "set_enable_reranker", "get_enable_reranker");

    ClassDB::bind_method(D_METHOD("set_reranker_model_path", "value"), &RagCorpusConfig::set_reranker_model_path);
    ClassDB::bind_method(D_METHOD("get_reranker_model_path"), &RagCorpusConfig::get_reranker_model_path);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "reranker_model_path", PROPERTY_HINT_FILE, "*.gguf"),
                 "set_reranker_model_path", "get_reranker_model_path");

    ClassDB::bind_method(D_METHOD("set_parser_mode", "value"), &RagCorpusConfig::set_parser_mode);
    ClassDB::bind_method(D_METHOD("get_parser_mode"), &RagCorpusConfig::get_parser_mode);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "parser_mode", PROPERTY_HINT_ENUM, "auto,text,markdown"),
                 "set_parser_mode", "get_parser_mode");

    ClassDB::bind_method(D_METHOD("set_supported_extensions", "value"), &RagCorpusConfig::set_supported_extensions);
    ClassDB::bind_method(D_METHOD("get_supported_extensions"), &RagCorpusConfig::get_supported_extensions);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "supported_extensions"), "set_supported_extensions",
                 "get_supported_extensions");
}

void RagCorpusConfig::set_storage_path(const String &path) { storage_path_ = path; }
String RagCorpusConfig::get_storage_path() const { return storage_path_; }

void RagCorpusConfig::set_chunk_size_tokens(int32_t value) { chunk_size_tokens_ = value > 0 ? value : 256; }
int32_t RagCorpusConfig::get_chunk_size_tokens() const { return chunk_size_tokens_; }

void RagCorpusConfig::set_chunk_overlap_tokens(int32_t value) { chunk_overlap_tokens_ = value >= 0 ? value : 0; }
int32_t RagCorpusConfig::get_chunk_overlap_tokens() const { return chunk_overlap_tokens_; }

void RagCorpusConfig::set_normalize_embeddings(bool value) { normalize_embeddings_ = value; }
bool RagCorpusConfig::get_normalize_embeddings() const { return normalize_embeddings_; }

void RagCorpusConfig::set_vector_metric(const String &value) { vector_metric_ = value; }
String RagCorpusConfig::get_vector_metric() const { return vector_metric_; }

void RagCorpusConfig::set_max_batch_texts(int32_t value) { max_batch_texts_ = value > 0 ? value : 8; }
int32_t RagCorpusConfig::get_max_batch_texts() const { return max_batch_texts_; }

void RagCorpusConfig::set_embedding_model_path(const String &value) { embedding_model_path_ = value; }
String RagCorpusConfig::get_embedding_model_path() const { return embedding_model_path_; }

void RagCorpusConfig::set_embedding_n_ctx(int32_t value) { embedding_n_ctx_ = value > 0 ? value : 2048; }
int32_t RagCorpusConfig::get_embedding_n_ctx() const { return embedding_n_ctx_; }

void RagCorpusConfig::set_embedding_n_threads(int32_t value) { embedding_n_threads_ = value; }
int32_t RagCorpusConfig::get_embedding_n_threads() const { return embedding_n_threads_; }

void RagCorpusConfig::set_enable_reranker(bool value) { enable_reranker_ = value; }
bool RagCorpusConfig::get_enable_reranker() const { return enable_reranker_; }

void RagCorpusConfig::set_reranker_model_path(const String &value) { reranker_model_path_ = value; }
String RagCorpusConfig::get_reranker_model_path() const { return reranker_model_path_; }

void RagCorpusConfig::set_parser_mode(const String &value) { parser_mode_ = value; }
String RagCorpusConfig::get_parser_mode() const { return parser_mode_; }

void RagCorpusConfig::set_supported_extensions(const PackedStringArray &value) { supported_extensions_ = value; }
PackedStringArray RagCorpusConfig::get_supported_extensions() const { return supported_extensions_; }

} // namespace godot
