#include <catch2/catch_test_macros.hpp>

#include "godot_llama/rag/types.hpp"

using namespace godot_llama::rag;

TEST_CASE("RAG type helpers are deterministic", "[rag][types]") {
    REQUIRE(parse_vector_metric("cosine").value() == VectorMetric::Cosine);
    REQUIRE(parse_vector_metric("dot").value() == VectorMetric::Dot);
    REQUIRE(parse_parser_mode("auto").value() == ParserMode::Auto);
    REQUIRE(parse_parser_mode("markdown").value() == ParserMode::Markdown);

    Metadata metadata = {{"topic", "  rag "}, {"lang", " en "}, {"topic", "override"}};
    metadata = canonicalize_metadata(std::move(metadata));
    REQUIRE(metadata.size() == 2);
    REQUIRE(metadata[0].key == "lang");
    REQUIRE(metadata[0].value == "en");
    REQUIRE(metadata[1].key == "topic");
    REQUIRE(metadata[1].value == "rag");

    REQUIRE(stable_hash_hex("abc") == stable_hash_hex("abc"));
    REQUIRE(make_source_version("hello world") == make_source_version("hello world"));
    REQUIRE(make_chunk_id("doc", "v1", 0, 10, 0) == make_chunk_id("doc", "v1", 0, 10, 0));
    REQUIRE(make_chunk_id("doc", "v1", 0, 10, 0) != make_chunk_id("doc", "v1", 0, 11, 0));
}
