#include "godot_llama/llama_multimodal_handle.hpp"

#include "godot_llama/llama_model_handle.hpp"

#include <filesystem>
#include <vector>
#include <system_error>

#if defined(GODOT_LLAMA_HAS_MTMD)
#include <mtmd.h>
#include <mtmd-helper.h>
#endif

namespace godot_llama {

LlamaMultimodalHandle::~LlamaMultimodalHandle() {
#if defined(GODOT_LLAMA_HAS_MTMD)
    if (ctx_ != nullptr) {
        mtmd_free(ctx_);
        ctx_ = nullptr;
    }
#endif
}

LlamaMultimodalHandle::LlamaMultimodalHandle(LlamaMultimodalHandle &&other) noexcept
        : ctx_(other.ctx_),
          media_marker_(std::move(other.media_marker_)),
          supports_vision_(other.supports_vision_),
          supports_audio_(other.supports_audio_),
          audio_sample_rate_hz_(other.audio_sample_rate_hz_) {
    other.ctx_ = nullptr;
    other.supports_vision_ = false;
    other.supports_audio_ = false;
    other.audio_sample_rate_hz_ = -1;
}

LlamaMultimodalHandle &LlamaMultimodalHandle::operator=(LlamaMultimodalHandle &&other) noexcept {
    if (this != &other) {
#if defined(GODOT_LLAMA_HAS_MTMD)
        if (ctx_ != nullptr) {
            mtmd_free(ctx_);
        }
#endif
        ctx_ = other.ctx_;
        media_marker_ = std::move(other.media_marker_);
        supports_vision_ = other.supports_vision_;
        supports_audio_ = other.supports_audio_;
        audio_sample_rate_hz_ = other.audio_sample_rate_hz_;

        other.ctx_ = nullptr;
        other.supports_vision_ = false;
        other.supports_audio_ = false;
        other.audio_sample_rate_hz_ = -1;
    }
    return *this;
}

Error LlamaMultimodalHandle::create(const std::shared_ptr<LlamaModelHandle> &model, const MultimodalConfig &config,
                                    LlamaMultimodalHandle &out) {
#if !defined(GODOT_LLAMA_HAS_MTMD)
    (void) model;
    (void) config;
    (void) out;
    return Error::make(ErrorCode::CapabilityUnavailable,
                       "This build does not include llama.cpp libmtmd support",
                       "Enable GODOT_LLAMA_ENABLE_MTMD to load multimodal projector files");
#else
    namespace fs = std::filesystem;

    if (!model || !model->is_loaded()) {
        return Error::make(ErrorCode::ModelLoadFailed, "Model must be loaded before initializing multimodal support");
    }

    if (config.mmproj_path.empty()) {
        return Error::make(ErrorCode::InvalidPath, "Multimodal projector path is empty");
    }

    std::error_code ec;
    if (!fs::exists(config.mmproj_path, ec)) {
        return Error::make(ErrorCode::InvalidPath,
                           "Multimodal projector file does not exist: " + config.mmproj_path);
    }

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = config.use_gpu;
    params.print_timings = config.print_timings;
    params.n_threads = config.n_threads;
    params.media_marker = config.media_marker.c_str();
    params.image_min_tokens = config.image_min_tokens;
    params.image_max_tokens = config.image_max_tokens;

    mtmd_context *ctx = mtmd_init_from_file(config.mmproj_path.c_str(), model->raw(), params);
    if (ctx == nullptr) {
        return Error::make(ErrorCode::CapabilityUnavailable,
                           "Failed to initialize multimodal projector from: " + config.mmproj_path,
                           "libmtmd initialization failed");
    }

    out.ctx_ = ctx;
    out.media_marker_ = config.media_marker.empty() ? std::string(kDefaultMediaMarker) : config.media_marker;
    out.supports_vision_ = mtmd_support_vision(ctx);
    out.supports_audio_ = mtmd_support_audio(ctx);
    out.audio_sample_rate_hz_ = mtmd_get_audio_sample_rate(ctx);
    return Error::make_ok();
#endif
}

bool LlamaMultimodalHandle::is_valid() const noexcept {
    return ctx_ != nullptr;
}

bool LlamaMultimodalHandle::supports_vision() const noexcept {
    return supports_vision_;
}

bool LlamaMultimodalHandle::supports_audio() const noexcept {
    return supports_audio_;
}

int32_t LlamaMultimodalHandle::audio_sample_rate_hz() const noexcept {
    return audio_sample_rate_hz_;
}

const std::string &LlamaMultimodalHandle::media_marker() const noexcept {
    return media_marker_;
}

Error LlamaMultimodalHandle::evaluate_prompt(llama_context *lctx, std::string_view prompt,
                                             std::span<const MultimodalInput> media_inputs, bool add_special,
                                             int32_t n_batch, bool logits_last, int32_t &out_n_past) const {
#if !defined(GODOT_LLAMA_HAS_MTMD)
    (void) lctx;
    (void) prompt;
    (void) media_inputs;
    (void) add_special;
    (void) n_batch;
    (void) logits_last;
    (void) out_n_past;
    return Error::make(ErrorCode::CapabilityUnavailable,
                       "This build does not include llama.cpp libmtmd support");
#else
    if (ctx_ == nullptr) {
        return Error::make(ErrorCode::CapabilityUnavailable,
                           "Multimodal projector is not initialized for this session");
    }

    if (lctx == nullptr) {
        return Error::make(ErrorCode::InvalidParameter, "Llama context is null");
    }

    if (prompt.empty()) {
        return Error::make(ErrorCode::InvalidParameter, "Multimodal prompt is empty");
    }

    if (media_inputs.empty()) {
        return Error::make(ErrorCode::InvalidParameter, "Multimodal request did not provide any media inputs");
    }

    mtmd::bitmaps bitmaps;
    bitmaps.entries.reserve(media_inputs.size());

    for (size_t i = 0; i < media_inputs.size(); ++i) {
        const auto &input = media_inputs[i];
        const std::string label =
                input.path.empty() ? ("media_inputs[" + std::to_string(i) + "]") : input.path;

        if (input.data.empty() && input.path.empty()) {
            return Error::make(ErrorCode::InvalidParameter,
                               "Multimodal input has neither 'data' nor 'path': " + label);
        }

        mtmd::bitmap bitmap(
                input.data.empty()
                        ? mtmd_helper_bitmap_init_from_file(const_cast<mtmd_context *>(ctx_), input.path.c_str())
                        : mtmd_helper_bitmap_init_from_buf(const_cast<mtmd_context *>(ctx_), input.data.data(),
                                                           input.data.size()));
        if (!bitmap.ptr) {
            return Error::make(ErrorCode::UnsupportedFormat,
                               "Failed to load multimodal input: " + label);
        }

        const bool is_audio = mtmd_bitmap_is_audio(bitmap.ptr.get());
        if (input.type == MultimodalInputType::Image && is_audio) {
            return Error::make(ErrorCode::UnsupportedFormat,
                               "Multimodal input was declared as image but decoded as audio: " + label);
        }
        if (input.type == MultimodalInputType::Audio && !is_audio) {
            return Error::make(ErrorCode::UnsupportedFormat,
                               "Multimodal input was declared as audio but decoded as image: " + label);
        }
        if (input.type == MultimodalInputType::Image && !supports_vision_) {
            return Error::make(ErrorCode::CapabilityUnavailable,
                               "Current multimodal projector does not support image input");
        }
        if (input.type == MultimodalInputType::Audio && !supports_audio_) {
            return Error::make(ErrorCode::CapabilityUnavailable,
                               "Current multimodal projector does not support audio input");
        }

        if (!input.id.empty()) {
            bitmap.set_id(input.id.c_str());
        }

        bitmaps.entries.push_back(std::move(bitmap));
    }

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    const std::string prompt_storage(prompt);
    mtmd_input_text text = {
            /* text = */ prompt_storage.c_str(),
            /* add_special = */ add_special,
            /* parse_special = */ true,
    };

    const auto bitmap_ptrs = bitmaps.c_ptr();
    auto **bitmap_ptr_data = bitmap_ptrs.empty() ? nullptr : const_cast<const mtmd_bitmap **>(bitmap_ptrs.data());
    const int32_t tokenize_result =
            mtmd_tokenize(const_cast<mtmd_context *>(ctx_), chunks.ptr.get(), &text, bitmap_ptr_data, bitmap_ptrs.size());
    if (tokenize_result == 1) {
        return Error::make(ErrorCode::InvalidParameter,
                           "Number of media inputs does not match the number of multimodal media markers in the prompt",
                           "Expected marker: " + media_marker_);
    }
    if (tokenize_result == 2) {
        return Error::make(ErrorCode::UnsupportedFormat,
                           "Failed to preprocess one or more multimodal inputs");
    }
    if (tokenize_result != 0) {
        return Error::make(ErrorCode::InternalError,
                           "mtmd_tokenize failed for multimodal prompt",
                           "status=" + std::to_string(tokenize_result));
    }

    llama_pos new_n_past = 0;
    const int32_t eval_result = mtmd_helper_eval_chunks(const_cast<mtmd_context *>(ctx_), lctx, chunks.ptr.get(), 0, 0,
                                                        n_batch, logits_last, &new_n_past);
    if (eval_result != 0) {
        return Error::make(ErrorCode::DecodeFailed,
                           "Failed to evaluate multimodal prompt",
                           "mtmd_helper_eval_chunks status=" + std::to_string(eval_result));
    }

    out_n_past = static_cast<int32_t>(new_n_past);
    return Error::make_ok();
#endif
}

} // namespace godot_llama
