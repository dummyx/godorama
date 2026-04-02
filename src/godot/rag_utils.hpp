#pragma once

#include "godot_llama/llama_params.hpp"
#include "godot_llama/rag/types.hpp"

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>

namespace godot {
class LlamaModelConfig;
}

namespace godot_llama::godot_rag {

[[nodiscard]] Error dictionary_to_metadata(const godot::Dictionary &dictionary, int32_t max_entries,
                                           int32_t max_key_bytes, int32_t max_value_bytes, rag::Metadata &out_metadata);
[[nodiscard]] rag::RetrievalOptions to_internal_retrieval_options(const godot::Dictionary &options);
[[nodiscard]] GenerateOptions to_internal_generate_options(const godot::Dictionary &options);
[[nodiscard]] ModelConfig to_internal_model_config(const godot::Ref<godot::Resource> &config);

[[nodiscard]] godot::Dictionary to_godot_dictionary(const rag::CorpusStats &stats);
[[nodiscard]] godot::Dictionary to_godot_dictionary(const rag::IngestStats &stats);
[[nodiscard]] godot::Dictionary to_godot_dictionary(const rag::RetrievalStats &stats);
[[nodiscard]] godot::Dictionary to_godot_dictionary(const rag::AnswerStats &stats);
[[nodiscard]] godot::Array retrieval_hits_to_array(const std::vector<rag::RetrievalHit> &hits);
[[nodiscard]] godot::Array citations_to_array(const std::vector<rag::Citation> &citations);

} // namespace godot_llama::godot_rag
