#include "godot_llama/rag/factories.hpp"

#include "godot_llama/utf8.hpp"

#include <algorithm>
#include <string_view>

namespace godot_llama::rag {
namespace {

struct Segment {
    std::string text;
    int64_t byte_start = 0;
    int64_t byte_end = 0;
    int32_t char_start = 0;
    int32_t char_end = 0;
    int32_t token_count = 0;
};

bool is_blank_line(std::string_view line) {
    for (const char ch : line) {
        if (ch != ' ' && ch != '\t' && ch != '\n') {
            return false;
        }
    }
    return true;
}

bool starts_with_marker(std::string_view line, std::string_view marker) {
    if (line.size() < marker.size()) {
        return false;
    }
    return std::equal(marker.begin(), marker.end(), line.begin());
}

bool is_heading_line(std::string_view line) {
    return !line.empty() && line[0] == '#';
}

bool is_list_line(std::string_view line) {
    return starts_with_marker(line, "- ") || starts_with_marker(line, "* ") || starts_with_marker(line, "+ ");
}

bool is_utf8_boundary(std::string_view text, size_t offset) {
    if (offset == 0 || offset >= text.size()) {
        return true;
    }
    const unsigned char byte = static_cast<unsigned char>(text[offset]);
    return (byte & 0xC0) != 0x80;
}

size_t previous_utf8_boundary(std::string_view text, size_t offset) {
    if (offset >= text.size()) {
        return text.size();
    }
    while (offset > 0 && !is_utf8_boundary(text, offset)) {
        --offset;
    }
    return offset;
}

Error count_tokens_safe(const TokenCounter &counter, std::string_view text, int32_t &out_count) {
    out_count = 0;
    if (text.empty()) {
        return Error::make_ok();
    }
    return counter.count_tokens(text, out_count);
}

Error emit_chunk(std::string_view source_id, std::string_view source_version, std::string_view title,
                 std::string_view source_path, const Metadata &metadata, int32_t chunk_index, const Segment &segment,
                 std::vector<ChunkRecord> &out_chunks) {
    if (segment.text.empty()) {
        return Error::make_ok();
    }

    ChunkRecord chunk;
    chunk.chunk_id = make_chunk_id(source_id, source_version, segment.byte_start, segment.byte_end, chunk_index);
    chunk.source_id = std::string(source_id);
    chunk.source_version = std::string(source_version);
    chunk.title = std::string(title);
    chunk.source_path = std::string(source_path);
    chunk.normalized_text = segment.text;
    chunk.display_text = segment.text;
    chunk.metadata = metadata;
    chunk.chunk_index = chunk_index;
    chunk.byte_start = segment.byte_start;
    chunk.byte_end = segment.byte_end;
    chunk.char_start = segment.char_start;
    chunk.char_end = segment.char_end;
    chunk.token_count = segment.token_count;
    out_chunks.push_back(std::move(chunk));
    return Error::make_ok();
}

Error split_segment_greedily(const Segment &segment, const TokenCounter &counter, int32_t max_tokens,
                             std::vector<Segment> &out_segments) {
    size_t offset = 0;
    int32_t char_offset = 0;
    const std::string_view text(segment.text);

    while (offset < text.size()) {
        size_t low = offset + 1;
        size_t high = text.size();
        size_t best = offset;

        while (low <= high) {
            const size_t mid_raw = low + ((high - low) / 2);
            const size_t mid = previous_utf8_boundary(text, mid_raw);
            if (mid <= offset) {
                low = mid_raw + 1;
                continue;
            }

            int32_t token_count = 0;
            const Error err = count_tokens_safe(counter, text.substr(offset, mid - offset), token_count);
            if (err) {
                return err;
            }

            if (token_count <= max_tokens) {
                best = mid;
                low = mid_raw + 1;
            } else {
                if (mid == 0) {
                    break;
                }
                high = mid - 1;
            }
        }

        if (best <= offset) {
            return Error::make(ErrorCode::BudgetExceeded, "Unable to split oversized segment within token budget");
        }

        Segment part;
        part.text = std::string(text.substr(offset, best - offset));
        part.byte_start = segment.byte_start + static_cast<int64_t>(offset);
        part.byte_end = segment.byte_start + static_cast<int64_t>(best);
        part.char_start = segment.char_start + char_offset;
        part.char_end = part.char_start + utf8::codepoint_count(part.text);
        part.token_count = 0;
        const Error err = count_tokens_safe(counter, part.text, part.token_count);
        if (err) {
            return err;
        }

        char_offset += part.char_end - part.char_start;
        offset = best;
        out_segments.push_back(std::move(part));
    }

    return Error::make_ok();
}

Error split_oversized_segment(const Segment &segment, const TokenCounter &counter, int32_t max_tokens,
                              std::vector<Segment> &out_segments) {
    const std::string_view text(segment.text);
    size_t cursor = 0;
    size_t piece_start = 0;
    int32_t char_cursor = segment.char_start;
    std::vector<Segment> candidate_pieces;

    auto flush_piece = [&](size_t end_offset) -> Error {
        if (end_offset <= piece_start) {
            return Error::make_ok();
        }

        Segment piece;
        piece.text = std::string(text.substr(piece_start, end_offset - piece_start));
        piece.byte_start = segment.byte_start + static_cast<int64_t>(piece_start);
        piece.byte_end = segment.byte_start + static_cast<int64_t>(end_offset);
        piece.char_start = char_cursor;
        piece.char_end = char_cursor + utf8::codepoint_count(piece.text);
        piece.token_count = 0;
        Error err = count_tokens_safe(counter, piece.text, piece.token_count);
        if (err) {
            return err;
        }
        char_cursor = piece.char_end;
        candidate_pieces.push_back(std::move(piece));
        return Error::make_ok();
    };

    while (cursor < text.size()) {
        const char ch = text[cursor];
        const bool boundary = ch == '\n' || ch == '.' || ch == '!' || ch == '?' || ch == ';';
        ++cursor;
        if (!boundary) {
            continue;
        }

        const Error err = flush_piece(cursor);
        if (err) {
            return err;
        }
        piece_start = cursor;
    }

    if (piece_start < text.size()) {
        const Error err = flush_piece(text.size());
        if (err) {
            return err;
        }
    }

    bool all_fit = !candidate_pieces.empty();
    for (const auto &piece : candidate_pieces) {
        if (piece.token_count > max_tokens) {
            all_fit = false;
            break;
        }
    }

    if (!all_fit) {
        return split_segment_greedily(segment, counter, max_tokens, out_segments);
    }

    out_segments.insert(out_segments.end(), candidate_pieces.begin(), candidate_pieces.end());
    return Error::make_ok();
}

std::vector<Segment> segment_document(const NormalizedDocument &document) {
    std::vector<Segment> segments;
    const std::string_view text(document.normalized_text);

    size_t cursor = 0;
    int32_t char_cursor = 0;
    bool in_code_fence = false;
    Segment current;
    current.byte_start = 0;
    current.char_start = 0;

    auto flush_current = [&]() {
        if (current.text.empty()) {
            current.byte_start = static_cast<int64_t>(cursor);
            current.char_start = char_cursor;
            return;
        }
        current.byte_end = current.byte_start + static_cast<int64_t>(current.text.size());
        current.char_end = current.char_start + utf8::codepoint_count(current.text);
        segments.push_back(current);
        current = {};
        current.byte_start = static_cast<int64_t>(cursor);
        current.char_start = char_cursor;
    };

    while (cursor < text.size()) {
        const size_t line_start = cursor;
        size_t line_end = text.find('\n', cursor);
        if (line_end == std::string_view::npos) {
            line_end = text.size();
        } else {
            ++line_end;
        }
        const std::string_view line = text.substr(line_start, line_end - line_start);
        const std::string_view line_trimmed = line.substr(0, line.empty() ? 0 : line.size() - (line.back() == '\n' ? 1 : 0));

        if (starts_with_marker(line_trimmed, "```")) {
            if (!in_code_fence) {
                flush_current();
                current.byte_start = static_cast<int64_t>(line_start);
                current.char_start = char_cursor;
                in_code_fence = true;
                current.text.append(line);
            } else {
                current.text.append(line);
                in_code_fence = false;
                flush_current();
            }
        } else if (in_code_fence) {
            current.text.append(line);
        } else if (is_blank_line(line)) {
            if (!current.text.empty()) {
                current.text.append(line);
                flush_current();
            }
        } else if (is_heading_line(line_trimmed) || is_list_line(line_trimmed)) {
            flush_current();
            current.text.assign(line);
            flush_current();
        } else {
            if (current.text.empty()) {
                current.byte_start = static_cast<int64_t>(line_start);
                current.char_start = char_cursor;
            }
            current.text.append(line);
        }

        char_cursor += utf8::codepoint_count(line);
        cursor = line_end;
    }

    flush_current();
    return segments;
}

class DeterministicChunker final : public Chunker {
public:
    [[nodiscard]] Error chunk(const NormalizedDocument &document, const TokenCounter &token_counter,
                              const ChunkingConfig &config, std::vector<ChunkRecord> &out_chunks) const override {
        out_chunks.clear();
        if (config.chunk_size_tokens <= 0) {
            return Error::make(ErrorCode::InvalidParameter, "chunk_size_tokens must be positive");
        }
        if (document.normalized_text.empty()) {
            return Error::make_ok();
        }

        std::vector<Segment> segments = segment_document(document);
        std::vector<Segment> expanded_segments;

        for (auto &segment : segments) {
            Error err = count_tokens_safe(token_counter, segment.text, segment.token_count);
            if (err) {
                return err;
            }
            if (segment.token_count <= config.chunk_size_tokens) {
                expanded_segments.push_back(std::move(segment));
                continue;
            }
            err = split_oversized_segment(segment, token_counter, config.chunk_size_tokens, expanded_segments);
            if (err) {
                return err;
            }
        }

        std::vector<Segment> current_segments;
        int32_t current_tokens = 0;
        int32_t chunk_index = 0;

        auto flush_chunk = [&](bool carry_overlap) -> Error {
            if (current_segments.empty()) {
                return Error::make_ok();
            }

            Segment merged;
            merged.byte_start = current_segments.front().byte_start;
            merged.char_start = current_segments.front().char_start;
            merged.byte_end = current_segments.back().byte_end;
            merged.char_end = current_segments.back().char_end;
            merged.token_count = 0;

            for (const auto &segment : current_segments) {
                merged.text.append(segment.text);
                merged.token_count += segment.token_count;
            }

            Error err = emit_chunk(document.source_id, document.source_version, document.title, document.source_path,
                                   document.metadata, chunk_index++, merged, out_chunks);
            if (err) {
                return err;
            }

            if (!carry_overlap || config.chunk_overlap_tokens <= 0) {
                current_segments.clear();
                current_tokens = 0;
                return Error::make_ok();
            }

        std::vector<Segment> overlap_segments;
        int32_t overlap_tokens = 0;
        for (auto it = current_segments.rbegin(); it != current_segments.rend(); ++it) {
            overlap_segments.push_back(*it);
            overlap_tokens += it->token_count;
            if (overlap_tokens >= config.chunk_overlap_tokens) {
                    break;
                }
            }

            std::reverse(overlap_segments.begin(), overlap_segments.end());
            current_segments = std::move(overlap_segments);
            current_tokens = 0;
            for (const auto &segment : current_segments) {
                current_tokens += segment.token_count;
            }
            return Error::make_ok();
        };

        for (const auto &segment : expanded_segments) {
            if (current_segments.empty()) {
                current_segments.push_back(segment);
                current_tokens = segment.token_count;
                continue;
            }

            if (current_tokens + segment.token_count <= config.chunk_size_tokens) {
                current_segments.push_back(segment);
                current_tokens += segment.token_count;
                continue;
            }

            Error err = flush_chunk(true);
            if (err) {
                return err;
            }

            if (!current_segments.empty() && current_tokens + segment.token_count <= config.chunk_size_tokens) {
                current_segments.push_back(segment);
                current_tokens += segment.token_count;
                continue;
            }

            current_segments.clear();
            current_tokens = 0;
            current_segments.push_back(segment);
            current_tokens = segment.token_count;
        }

        return flush_chunk(false);
    }
};

} // namespace

std::unique_ptr<Chunker> make_deterministic_chunker() {
    return std::make_unique<DeterministicChunker>();
}

} // namespace godot_llama::rag
