#pragma once

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

namespace godot {

class RagCorpusConfig : public Resource {
    GDCLASS(RagCorpusConfig, Resource)

public:
    RagCorpusConfig();
    ~RagCorpusConfig() override = default;

    void set_storage_path(const String &path);
    String get_storage_path() const;

    void set_chunk_size_tokens(int32_t value);
    int32_t get_chunk_size_tokens() const;

    void set_chunk_overlap_tokens(int32_t value);
    int32_t get_chunk_overlap_tokens() const;

    void set_normalize_embeddings(bool value);
    bool get_normalize_embeddings() const;

    void set_max_batch_texts(int32_t value);
    int32_t get_max_batch_texts() const;

    void set_embedding_model_path(const String &value);
    String get_embedding_model_path() const;

    void set_embedding_n_ctx(int32_t value);
    int32_t get_embedding_n_ctx() const;

    void set_embedding_n_threads(int32_t value);
    int32_t get_embedding_n_threads() const;

    void set_enable_reranker(bool value);
    bool get_enable_reranker() const;

    void set_reranker_model_path(const String &value);
    String get_reranker_model_path() const;

    void set_parser_mode(const String &value);
    String get_parser_mode() const;

    void set_supported_extensions(const PackedStringArray &value);
    PackedStringArray get_supported_extensions() const;

protected:
    static void _bind_methods();

private:
    String storage_path_;
    int32_t chunk_size_tokens_ = 256;
    int32_t chunk_overlap_tokens_ = 32;
    bool normalize_embeddings_ = true;
    int32_t max_batch_texts_ = 8;
    String embedding_model_path_;
    int32_t embedding_n_ctx_ = 2048;
    int32_t embedding_n_threads_ = -1;
    bool enable_reranker_ = false;
    String reranker_model_path_;
    String parser_mode_ = "auto";
    PackedStringArray supported_extensions_;
};

} // namespace godot
