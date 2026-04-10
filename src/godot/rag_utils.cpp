#include "rag_utils.hpp"

#include "llama_model_config.hpp"

#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace godot_llama::godot_rag {
namespace {

std::string to_utf8_string(const godot::String &value) {
    const auto utf8 = value.utf8();
    return {utf8.get_data(), static_cast<size_t>(utf8.length())};
}

godot::Dictionary metadata_to_dictionary(const rag::Metadata &metadata) {
    godot::Dictionary dictionary;
    for (const auto &entry : metadata) {
        dictionary[godot::String(entry.key.c_str())] = godot::String(entry.value.c_str());
    }
    return dictionary;
}

} // namespace

Error dictionary_to_metadata(const godot::Dictionary &dictionary, int32_t max_entries, int32_t max_key_bytes,
                             int32_t max_value_bytes, rag::Metadata &out_metadata) {
    out_metadata.clear();
    if (dictionary.size() > max_entries) {
        return Error::make(ErrorCode::MetadataTooLarge, "Too many metadata entries");
    }

    godot::Array keys = dictionary.keys();
    out_metadata.reserve(static_cast<size_t>(keys.size()));
    for (int64_t index = 0; index < keys.size(); ++index) {
        const godot::String key = keys[index];
        const godot::String value = dictionary[key];
        const std::string key_utf8 = to_utf8_string(key);
        const std::string value_utf8 = to_utf8_string(value);
        if (static_cast<int32_t>(key_utf8.size()) > max_key_bytes ||
            static_cast<int32_t>(value_utf8.size()) > max_value_bytes) {
            return Error::make(ErrorCode::MetadataTooLarge, "Metadata entry exceeds configured limits", key_utf8);
        }
        out_metadata.push_back({key_utf8, value_utf8});
    }
    out_metadata = rag::canonicalize_metadata(std::move(out_metadata));
    return Error::make_ok();
}

rag::RetrievalOptions to_internal_retrieval_options(const godot::Dictionary &options) {
    rag::RetrievalOptions result;

    if (options.has("top_k")) {
        result.top_k = static_cast<int32_t>(options["top_k"]);
    }
    if (options.has("candidate_k")) {
        result.candidate_k = static_cast<int32_t>(options["candidate_k"]);
    }
    if (options.has("score_threshold")) {
        result.score_threshold = static_cast<float>(static_cast<double>(options["score_threshold"]));
    }
    if (options.has("max_context_chunks")) {
        result.max_context_chunks = static_cast<int32_t>(options["max_context_chunks"]);
    }
    if (options.has("max_context_tokens")) {
        result.max_context_tokens = static_cast<int32_t>(options["max_context_tokens"]);
    }
    if (options.has("use_mmr")) {
        result.use_mmr = static_cast<bool>(options["use_mmr"]);
    }
    if (options.has("use_reranker")) {
        result.use_reranker = static_cast<bool>(options["use_reranker"]);
    }
    if (options.has("metadata_filter")) {
        rag::Metadata metadata_filter;
        const godot::Dictionary metadata_dictionary = options["metadata_filter"];
        if (!dictionary_to_metadata(metadata_dictionary, 32, 64, 256, metadata_filter)) {
            result.metadata_filter = std::move(metadata_filter);
        }
    }
    if (options.has("source_ids")) {
        const godot::Array array = options["source_ids"];
        for (int64_t index = 0; index < array.size(); ++index) {
            result.source_ids.push_back(to_utf8_string(array[index]));
        }
    }
    if (options.has("exclude_source_ids")) {
        const godot::Array array = options["exclude_source_ids"];
        for (int64_t index = 0; index < array.size(); ++index) {
            result.exclude_source_ids.push_back(to_utf8_string(array[index]));
        }
    }

    return result;
}

GenerateOptions to_internal_generate_options(const godot::Dictionary &options) {
    GenerateOptions result;
    if (options.has("max_tokens")) {
        result.max_tokens = static_cast<int32_t>(options["max_tokens"]);
    }
    if (options.has("temperature")) {
        result.temperature = static_cast<float>(static_cast<double>(options["temperature"]));
    }
    if (options.has("top_p")) {
        result.top_p = static_cast<float>(static_cast<double>(options["top_p"]));
    }
    if (options.has("top_k")) {
        result.top_k = static_cast<int32_t>(options["top_k"]);
    }
    if (options.has("min_p")) {
        result.min_p = static_cast<float>(static_cast<double>(options["min_p"]));
    }
    if (options.has("repeat_penalty")) {
        result.repeat_penalty = static_cast<float>(static_cast<double>(options["repeat_penalty"]));
    }
    if (options.has("seed_override")) {
        result.seed_override = static_cast<uint32_t>(static_cast<int64_t>(options["seed_override"]));
    }
    if (options.has("stop")) {
        const godot::Array array = options["stop"];
        for (int64_t index = 0; index < array.size(); ++index) {
            result.stop.push_back(to_utf8_string(array[index]));
        }
    }
    return result;
}

ModelConfig to_internal_model_config(const godot::Ref<godot::Resource> &config) {
    ModelConfig internal;

    godot::LlamaModelConfig *model_config = godot::Object::cast_to<godot::LlamaModelConfig>(config.ptr());
    if (!model_config) {
        return internal;
    }

    internal.model_path = to_utf8_string(model_config->get_model_path());
    internal.n_ctx = model_config->get_n_ctx();
    internal.n_threads = model_config->get_n_threads();
    internal.n_batch = model_config->get_n_batch();
    internal.n_gpu_layers = model_config->get_n_gpu_layers();
    internal.seed = static_cast<uint32_t>(model_config->get_seed());
    internal.use_mmap = model_config->get_use_mmap();
    internal.use_mlock = model_config->get_use_mlock();
    internal.embeddings_enabled = model_config->get_embeddings_enabled();
    internal.chat_template_override = to_utf8_string(model_config->get_chat_template_override());
    return internal;
}

godot::Dictionary to_godot_dictionary(const rag::CorpusStats &stats) {
    godot::Dictionary dictionary;
    dictionary["schema_version"] = stats.schema_version;
    dictionary["source_count"] = static_cast<int64_t>(stats.source_count);
    dictionary["chunk_count"] = static_cast<int64_t>(stats.chunk_count);
    dictionary["embedding_model_fingerprint"] = godot::String(stats.embedding_model_fingerprint.c_str());
    dictionary["embedding_dimensions"] = stats.embedding_dimensions;
    dictionary["embedding_normalized"] = stats.embedding_normalized;
    dictionary["vector_metric"] = godot::String(rag::vector_metric_name(stats.vector_metric));
    dictionary["stale_embeddings"] = stats.stale_embeddings;
    return dictionary;
}

godot::Dictionary to_godot_dictionary(const rag::IngestStats &stats) {
    godot::Dictionary dictionary;
    dictionary["source_id"] = godot::String(stats.source_id.c_str());
    dictionary["source_version"] = godot::String(stats.source_version.c_str());
    dictionary["chunks_written"] = stats.chunks_written;
    dictionary["chunks_reused"] = stats.chunks_reused;
    dictionary["chunks_deleted"] = stats.chunks_deleted;
    dictionary["embeddings_generated"] = stats.embeddings_generated;
    return dictionary;
}

godot::Dictionary to_godot_dictionary(const rag::RetrievalStats &stats) {
    godot::Dictionary dictionary;
    dictionary["query_token_count"] = stats.query_token_count;
    dictionary["scanned_chunks"] = stats.scanned_chunks;
    dictionary["candidate_chunks"] = stats.candidate_chunks;
    dictionary["filtered_chunks"] = stats.filtered_chunks;
    dictionary["deduplicated_chunks"] = stats.deduplicated_chunks;
    dictionary["returned_chunks"] = stats.returned_chunks;
    dictionary["ann_fallback_used"] = stats.ann_fallback_used;
    dictionary["reranker_used"] = stats.reranker_used;
    dictionary["search_mode"] = godot::String(stats.search_mode.c_str());
    dictionary["reranker_status"] = godot::String(stats.reranker_status.c_str());

    godot::Array truncated;
    for (const auto &chunk_id : stats.truncated_chunk_ids) {
        truncated.push_back(godot::String(chunk_id.c_str()));
    }
    dictionary["truncated_chunk_ids"] = truncated;
    return dictionary;
}

godot::Dictionary to_godot_dictionary(const rag::AnswerStats &stats) {
    godot::Dictionary dictionary;
    dictionary["retrieval"] = to_godot_dictionary(stats.retrieval);
    dictionary["packed_chunks"] = stats.packed_chunks;
    dictionary["packed_context_tokens"] = stats.packed_context_tokens;
    dictionary["truncated_chunks"] = stats.truncated_chunks;
    dictionary["abstained"] = stats.abstained;
    dictionary["prompt_style"] = godot::String(stats.prompt_style.c_str());
    return dictionary;
}

godot::Array retrieval_hits_to_array(const std::vector<rag::RetrievalHit> &hits) {
    godot::Array array;
    for (const auto &hit : hits) {
        godot::Dictionary dictionary;
        dictionary["chunk_id"] = godot::String(hit.chunk_id.c_str());
        dictionary["source_id"] = godot::String(hit.source_id.c_str());
        dictionary["title"] = godot::String(hit.title.c_str());
        dictionary["source_path"] = godot::String(hit.source_path.c_str());
        dictionary["excerpt"] = godot::String(hit.excerpt.c_str());
        dictionary["metadata"] = metadata_to_dictionary(hit.metadata);
        dictionary["raw_score"] = hit.raw_score;
        dictionary["relevance_score"] = hit.relevance_score;
        if (hit.rerank_score.has_value()) {
            dictionary["rerank_score"] = *hit.rerank_score;
        }
        dictionary["byte_start"] = static_cast<int64_t>(hit.byte_start);
        dictionary["byte_end"] = static_cast<int64_t>(hit.byte_end);
        dictionary["char_start"] = hit.char_start;
        dictionary["char_end"] = hit.char_end;
        dictionary["token_count"] = hit.token_count;
        array.push_back(dictionary);
    }
    return array;
}

godot::Array citations_to_array(const std::vector<rag::Citation> &citations) {
    godot::Array array;
    for (const auto &citation : citations) {
        godot::Dictionary dictionary;
        dictionary["chunk_id"] = godot::String(citation.chunk_id.c_str());
        dictionary["source_id"] = godot::String(citation.source_id.c_str());
        dictionary["title"] = godot::String(citation.title.c_str());
        dictionary["source_path"] = godot::String(citation.source_path.c_str());
        dictionary["byte_start"] = static_cast<int64_t>(citation.byte_start);
        dictionary["byte_end"] = static_cast<int64_t>(citation.byte_end);
        dictionary["char_start"] = citation.char_start;
        dictionary["char_end"] = citation.char_end;
        dictionary["excerpt"] = godot::String(citation.excerpt.c_str());
        array.push_back(dictionary);
    }
    return array;
}

} // namespace godot_llama::godot_rag
