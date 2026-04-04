#include "godot_llama/llama_lora_adapter_handle.hpp"

#include <llama.h>

#include <cmath>
#include <filesystem>
#include <system_error>

namespace godot_llama {

LlamaLoraAdapterHandle::~LlamaLoraAdapterHandle() {
    if (adapter_ != nullptr) {
        llama_adapter_lora_free(adapter_);
        adapter_ = nullptr;
    }
}

LlamaLoraAdapterHandle::LlamaLoraAdapterHandle(LlamaLoraAdapterHandle &&other) noexcept
        : adapter_(other.adapter_),
          path_(std::move(other.path_)),
          scale_(other.scale_) {
    other.adapter_ = nullptr;
    other.scale_ = 1.0f;
}

LlamaLoraAdapterHandle &LlamaLoraAdapterHandle::operator=(LlamaLoraAdapterHandle &&other) noexcept {
    if (this != &other) {
        if (adapter_ != nullptr) {
            llama_adapter_lora_free(adapter_);
        }
        adapter_ = other.adapter_;
        path_ = std::move(other.path_);
        scale_ = other.scale_;
        other.adapter_ = nullptr;
        other.scale_ = 1.0f;
    }
    return *this;
}

Error LlamaLoraAdapterHandle::load(llama_model *model, const LoraAdapterConfig &config, LlamaLoraAdapterHandle &out) {
    namespace fs = std::filesystem;

    if (model == nullptr) {
        return Error::make(ErrorCode::ModelLoadFailed, "Model must be loaded before loading LoRA adapters");
    }

    if (config.path.empty()) {
        return Error::make(ErrorCode::InvalidPath, "LoRA adapter path is empty");
    }

    if (!std::isfinite(config.scale)) {
        return Error::make(ErrorCode::InvalidParameter, "LoRA adapter scale must be finite");
    }

    std::error_code ec;
    if (!fs::exists(config.path, ec)) {
        return Error::make(ErrorCode::InvalidPath, "LoRA adapter file does not exist: " + config.path);
    }

    llama_adapter_lora *adapter = llama_adapter_lora_init(model, config.path.c_str());
    if (adapter == nullptr) {
        return Error::make(ErrorCode::ModelLoadFailed, "Failed to load LoRA adapter from: " + config.path);
    }

    out.adapter_ = adapter;
    out.path_ = config.path;
    out.scale_ = config.scale;
    return Error::make_ok();
}

bool LlamaLoraAdapterHandle::is_valid() const noexcept {
    return adapter_ != nullptr;
}

llama_adapter_lora *LlamaLoraAdapterHandle::raw() const noexcept {
    return adapter_;
}

float LlamaLoraAdapterHandle::scale() const noexcept {
    return scale_;
}

const std::string &LlamaLoraAdapterHandle::path() const noexcept {
    return path_;
}

} // namespace godot_llama
