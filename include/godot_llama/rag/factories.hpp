#pragma once

#include "godot_llama/rag/interfaces.hpp"

#include <memory>
#include <vector>

namespace godot_llama::rag {

[[nodiscard]] Error make_sqlite_corpus_store(const CorpusConfig &config, std::unique_ptr<CorpusStore> &out_store);
[[nodiscard]] std::unique_ptr<Chunker> make_deterministic_chunker();
[[nodiscard]] Error make_llama_embedder(const CorpusConfig &config, std::unique_ptr<Embedder> &out_embedder);
[[nodiscard]] std::unique_ptr<Retriever> make_dense_retriever();
[[nodiscard]] std::unique_ptr<Reranker> make_noop_reranker(const char *status_name = "disabled");
[[nodiscard]] std::unique_ptr<ContextPacker> make_grounded_context_packer();

} // namespace godot_llama::rag
