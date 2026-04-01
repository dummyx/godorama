#include "godot_llama/llama_sampler_handle.hpp"

#include <llama.h>

namespace godot_llama {

LlamaSamplerHandle::~LlamaSamplerHandle() {
    if (chain_) {
        llama_sampler_free(chain_);
        chain_ = nullptr;
    }
}

LlamaSamplerHandle::LlamaSamplerHandle(LlamaSamplerHandle &&other) noexcept : chain_(other.chain_) {
    other.chain_ = nullptr;
}

LlamaSamplerHandle &LlamaSamplerHandle::operator=(LlamaSamplerHandle &&other) noexcept {
    if (this != &other) {
        if (chain_) {
            llama_sampler_free(chain_);
        }
        chain_ = other.chain_;
        other.chain_ = nullptr;
    }
    return *this;
}

void LlamaSamplerHandle::init(const GenerateOptions &opts, const llama_vocab *vocab) {
    if (chain_) {
        llama_sampler_free(chain_);
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    chain_ = llama_sampler_chain_init(sparams);

    // Repetition penalty
    if (opts.repeat_penalty != 1.0f) {
        llama_sampler_chain_add(chain_, llama_sampler_init_penalties(opts.repeat_last_n, opts.repeat_penalty,
                                                                     0.0f, // frequency penalty
                                                                     0.0f  // presence penalty
                                                                     ));
    }

    // Top-K
    if (opts.top_k > 0) {
        llama_sampler_chain_add(chain_, llama_sampler_init_top_k(opts.top_k));
    }

    // Top-P
    if (opts.top_p < 1.0f && opts.top_p > 0.0f) {
        llama_sampler_chain_add(chain_, llama_sampler_init_top_p(opts.top_p, 1));
    }

    // Min-P
    if (opts.min_p > 0.0f && opts.min_p < 1.0f) {
        llama_sampler_chain_add(chain_, llama_sampler_init_min_p(opts.min_p, 1));
    }

    // Temperature
    if (opts.temperature > 0.0f) {
        llama_sampler_chain_add(chain_, llama_sampler_init_temp(opts.temperature));
    }

    // Final distribution sampler
    uint32_t seed = opts.seed_override.value_or(0xFFFFFFFF);
    if (opts.temperature <= 0.0f) {
        llama_sampler_chain_add(chain_, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain_, llama_sampler_init_dist(seed));
    }
}

void LlamaSamplerHandle::reset() noexcept {
    if (chain_) {
        llama_sampler_reset(chain_);
    }
}

int32_t LlamaSamplerHandle::sample(::llama_context *ctx, int32_t idx) {
    if (!chain_ || !ctx) {
        return -1;
    }
    return llama_sampler_sample(chain_, ctx, idx);
}

void LlamaSamplerHandle::accept(int32_t token) noexcept {
    if (chain_) {
        llama_sampler_accept(chain_, token);
    }
}

bool LlamaSamplerHandle::is_valid() const noexcept {
    return chain_ != nullptr;
}

} // namespace godot_llama
