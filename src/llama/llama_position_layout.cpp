#include "godot_llama/llama_position_layout.hpp"

#include <algorithm>

namespace godot_llama {
namespace {

Error expand_base_positions(std::span<const int32_t> base_positions, int32_t n_tokens, int32_t n_pos_per_embd,
                            std::vector<int32_t> &out) {
    if (n_tokens < 0 || n_pos_per_embd <= 0) {
        return Error::make(ErrorCode::InvalidParameter, "Position layout dimensions must be positive");
    }

    const size_t expected_token_positions = static_cast<size_t>(n_tokens);
    if (!base_positions.empty() && base_positions.size() != expected_token_positions) {
        return Error::make(ErrorCode::InvalidParameter,
                           "Base position array length must match sequence_length");
    }

    out.clear();
    if (n_tokens == 0) {
        return Error::make_ok();
    }

    if (n_pos_per_embd == 1) {
        out.resize(expected_token_positions);
        for (int32_t index = 0; index < n_tokens; ++index) {
            out[static_cast<size_t>(index)] =
                    base_positions.empty() ? index : base_positions[static_cast<size_t>(index)];
        }
        return Error::make_ok();
    }

    if (n_pos_per_embd == 3 || n_pos_per_embd == 4) {
        out.resize(expected_token_positions * static_cast<size_t>(n_pos_per_embd));
        for (int32_t index = 0; index < n_tokens; ++index) {
            const int32_t base = base_positions.empty() ? index : base_positions[static_cast<size_t>(index)];
            out[static_cast<size_t>(index)] = base;
            out[static_cast<size_t>(n_tokens + index)] = base;
            out[static_cast<size_t>((n_tokens * 2) + index)] = base;
            if (n_pos_per_embd == 4) {
                out[static_cast<size_t>((n_tokens * 3) + index)] = 0;
            }
        }
        return Error::make_ok();
    }

    return Error::make(ErrorCode::CapabilityUnavailable,
                       "Automatic position expansion only supports models with 1, 3, or 4 position components");
}

Error build_default_layout(int32_t n_tokens, int32_t n_pos_per_embd, std::vector<int32_t> &out) {
    return expand_base_positions({}, n_tokens, n_pos_per_embd, out);
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

    if (n_pos_per_embd == 3 || n_pos_per_embd == 4) {
        return expand_base_positions(positions, n_tokens, n_pos_per_embd, out);
    }

    return Error::make(ErrorCode::CapabilityUnavailable,
                       "Explicit base positions only support models with 1, 3, or 4 position components");
}

} // namespace godot_llama
