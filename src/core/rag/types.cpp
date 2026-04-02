#include "godot_llama/rag/types.hpp"

#include "godot_llama/utf8.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>

namespace godot_llama::rag {
namespace {

std::string trim_ascii(std::string_view value) {
    size_t begin = 0;
    while (begin < value.size() && (value[begin] == ' ' || value[begin] == '\t' || value[begin] == '\n' || value[begin] == '\r')) {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin &&
           (value[end - 1] == ' ' || value[end - 1] == '\t' || value[end - 1] == '\n' || value[end - 1] == '\r')) {
        --end;
    }

    return std::string(value.substr(begin, end - begin));
}

} // namespace

const char *vector_metric_name(VectorMetric metric) noexcept {
    switch (metric) {
    case VectorMetric::Cosine:
        return "cosine";
    case VectorMetric::Dot:
        return "dot";
    }
    return "cosine";
}

std::optional<VectorMetric> parse_vector_metric(std::string_view value) noexcept {
    if (value == "cosine") {
        return VectorMetric::Cosine;
    }
    if (value == "dot") {
        return VectorMetric::Dot;
    }
    return std::nullopt;
}

const char *parser_mode_name(ParserMode mode) noexcept {
    switch (mode) {
    case ParserMode::Auto:
        return "auto";
    case ParserMode::Text:
        return "text";
    case ParserMode::Markdown:
        return "markdown";
    }
    return "auto";
}

std::optional<ParserMode> parse_parser_mode(std::string_view value) noexcept {
    if (value == "auto") {
        return ParserMode::Auto;
    }
    if (value == "text") {
        return ParserMode::Text;
    }
    if (value == "markdown") {
        return ParserMode::Markdown;
    }
    return std::nullopt;
}

std::string canonicalize_metadata_value(std::string_view value) {
    std::string trimmed = trim_ascii(value);
    std::string canonical;
    canonical.reserve(trimmed.size());

    bool previous_space = false;
    for (char ch : trimmed) {
        const bool is_space = ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
        if (is_space) {
            if (!previous_space) {
                canonical.push_back(' ');
            }
            previous_space = true;
            continue;
        }

        canonical.push_back(ch);
        previous_space = false;
    }

    return canonical;
}

Metadata canonicalize_metadata(Metadata metadata) {
    for (auto &entry : metadata) {
        entry.key = trim_ascii(entry.key);
        entry.value = canonicalize_metadata_value(entry.value);
    }

    metadata.erase(std::remove_if(metadata.begin(), metadata.end(),
                                  [](const MetadataEntry &entry) { return entry.key.empty(); }),
                   metadata.end());

    std::sort(metadata.begin(), metadata.end(), [](const MetadataEntry &lhs, const MetadataEntry &rhs) {
        return lhs.key < rhs.key;
    });

    metadata.erase(std::unique(metadata.begin(), metadata.end(),
                               [](const MetadataEntry &lhs, const MetadataEntry &rhs) { return lhs.key == rhs.key; }),
                   metadata.end());

    return metadata;
}

std::optional<std::string> metadata_lookup(const Metadata &metadata, std::string_view key) {
    const auto found = std::lower_bound(metadata.begin(), metadata.end(), key,
                                        [](const MetadataEntry &entry, std::string_view value) {
                                            return entry.key < value;
                                        });
    if (found == metadata.end() || found->key != key) {
        return std::nullopt;
    }
    return found->value;
}

bool metadata_matches(const Metadata &metadata, const Metadata &filter) {
    for (const auto &entry : filter) {
        const auto value = metadata_lookup(metadata, entry.key);
        if (!value || *value != entry.value) {
            return false;
        }
    }
    return true;
}

std::string stable_hash_hex(std::string_view value) noexcept {
    constexpr uint64_t offset = 14695981039346656037ull;
    constexpr uint64_t prime = 1099511628211ull;

    uint64_t hash = offset;
    for (const unsigned char ch : value) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= prime;
    }

    std::array<char, 17> buffer{};
    snprintf(buffer.data(), buffer.size(), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(buffer.data());
}

std::string make_source_version(std::string_view normalized_text) noexcept {
    return stable_hash_hex(normalized_text);
}

std::string make_chunk_id(std::string_view source_id, std::string_view source_version, int64_t byte_start,
                          int64_t byte_end, int32_t chunk_index) noexcept {
    std::string seed;
    seed.reserve(source_id.size() + source_version.size() + 64);
    seed.append(source_id);
    seed.push_back('|');
    seed.append(source_version);
    seed.push_back('|');
    seed.append(std::to_string(byte_start));
    seed.push_back('|');
    seed.append(std::to_string(byte_end));
    seed.push_back('|');
    seed.append(std::to_string(chunk_index));
    return stable_hash_hex(seed);
}

std::string utc_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::tm utc_time{};
#if defined(_WIN32)
    gmtime_s(&utc_time, &time);
#else
    gmtime_r(&time, &utc_time);
#endif

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_time);
    return std::string(buffer);
}

} // namespace godot_llama::rag
