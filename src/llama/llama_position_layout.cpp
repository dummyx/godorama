#include "godot_llama/llama_position_layout.hpp"

#include <algorithm>

namespace godot_llama {
namespace {

Error build_default_layout(int32_t n_tokens, int32_t n_pos_per_embd, std::vector<int32_t> &out) {
    if (n_tokens < 0 || n_pos_per_embd <= 0) {
        return Error::make(ErrorCode::InvalidParameter, "Position layout dimensions must be positive");
    }

    out.clear();
    if (n_tokens == 0) {
        return Error::make_ok();
    }

    if (n_pos_per_embd == 1) {
        out.resize(static_cast<size_t>(n_tokens));
        for (int32_t index = 0; index < n_tokens; ++index) {
            out[static_cast<size_t>(index)] = index;
        }
        return Error::make_ok();
    }

    if (n_pos_per_embd == 4) {
        out.resize(static_cast<size_t>(n_tokens) * 4u);
        for (int32_t index = 0; index < n_tokens; ++index) {
            const int32_t base = index;
            out[static_cast<size_t>(index)] = base;
            out[static_cast<size_t>(n_tokens + index)] = base;
            out[static_cast<size_t>((n_tokens * 2) + index)] = base;
            out[static_cast<size_t>((n_tokens * 3) + index)] = 0;
        }
        return Error::make_ok();
    }

    return Error::make(ErrorCode::CapabilityUnavailable,
                       "Automatic position expansion only supports models with 1 or 4 position components");
}

} // namespace

Error normalize_position_layout(std::span<const int32_t> positions, int32_t n_tokens, int32_t n_pos_per_embd,
                                std::vector<int32_t> &out) {
    if (n_tokens < 0 || n_pos_per_embd <= 0) {
        return Error::make(ErrorCode::InvalidParameter, "Position layout dimensions must be positive");
    }

    if (positions.empty()) {
        return build_default_layout(n_tokens, n_pos_per_embd, out);
    }

    const size_t expected_token_positions = static_cast<size_t>(n_tokens);
    const size_t expected_component_positions = expected_token_positions * static_cast<size_t>(n_pos_per_embd);

    if (positions.size() == expected_component_positions) {
        out.assign(positions.begin(), positions.end());
        return Error::make_ok();
    }

    if (positions.size() != expected_token_positions) {
        return Error::make(
                ErrorCode::InvalidParameter,
                "Position array length must match either sequence_length or sequence_length * n_pos_per_embd");
    }

    if (n_pos_per_embd == 1) {
        out.assign(positions.begin(), positions.end());
        return Error::make_ok();
    }

    if (n_pos_per_embd == 4) {
        out.resize(expected_component_positions);
        for (int32_t index = 0; index < n_tokens; ++index) {
            const int32_t base = positions[static_cast<size_t>(index)];
            out[static_cast<size_t>(index)] = base;
            out[static_cast<size_t>(n_tokens + index)] = base;
            out[static_cast<size_t>((n_tokens * 2) + index)] = base;
            out[static_cast<size_t>((n_tokens * 3) + index)] = 0;
        }
        return Error::make_ok();
    }

    return Error::make(ErrorCode::CapabilityUnavailable,
                       "Explicit base positions only support models with 1 or 4 position components");
}

} // namespace godot_llama
