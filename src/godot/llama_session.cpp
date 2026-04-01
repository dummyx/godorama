#include "llama_session.hpp"

#include "godot_llama/llama_model_handle.hpp"
#include "llama_model_config.hpp"

#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

LlamaSession::LlamaSession() : worker_(std::make_unique<godot_llama::InferenceWorker>()) {}

LlamaSession::~LlamaSession() {
    close();
}

void LlamaSession::_bind_methods() {
    ClassDB::bind_method(D_METHOD("open", "config"), &LlamaSession::open);
    ClassDB::bind_method(D_METHOD("close"), &LlamaSession::close);
    ClassDB::bind_method(D_METHOD("is_open"), &LlamaSession::is_open);
    ClassDB::bind_method(D_METHOD("generate_async", "prompt", "options"), &LlamaSession::generate_async,
                         DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("cancel", "request_id"), &LlamaSession::cancel);
    ClassDB::bind_method(D_METHOD("tokenize", "text", "add_bos", "special"), &LlamaSession::tokenize, DEFVAL(false),
                         DEFVAL(false));
    ClassDB::bind_method(D_METHOD("detokenize", "tokens"), &LlamaSession::detokenize);
    ClassDB::bind_method(D_METHOD("embed", "text"), &LlamaSession::embed);
    ClassDB::bind_method(D_METHOD("poll"), &LlamaSession::poll);

    ADD_SIGNAL(MethodInfo("opened"));
    ADD_SIGNAL(MethodInfo("token_emitted", PropertyInfo(Variant::INT, "request_id"),
                          PropertyInfo(Variant::STRING, "token_text"), PropertyInfo(Variant::INT, "token_id")));
    ADD_SIGNAL(MethodInfo("completed", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::STRING, "text"),
                          PropertyInfo(Variant::DICTIONARY, "stats")));
    ADD_SIGNAL(MethodInfo("failed", PropertyInfo(Variant::INT, "request_id"), PropertyInfo(Variant::INT, "error_code"),
                          PropertyInfo(Variant::STRING, "error_message"), PropertyInfo(Variant::STRING, "details")));
    ADD_SIGNAL(MethodInfo("cancelled", PropertyInfo(Variant::INT, "request_id")));
}

int LlamaSession::open(const Ref<Resource> &config) {
    if (is_open_) {
        UtilityFunctions::push_error("LlamaSession: already open, call close() first");
        return static_cast<int>(godot_llama::ErrorCode::AlreadyOpen);
    }

    if (config.is_null()) {
        UtilityFunctions::push_error("LlamaSession: config is null");
        return static_cast<int>(godot_llama::ErrorCode::InvalidParameter);
    }

    auto internal_config = to_internal_config(config);

    // Load model (this is blocking — for production use, consider loading on a thread)
    std::shared_ptr<godot_llama::LlamaModelHandle> model;
    auto load_err = godot_llama::LlamaModelHandle::load(internal_config, model);
    if (load_err) {
        UtilityFunctions::push_error(String("LlamaSession: ") + load_err.message.c_str());
        emit_signal("failed", 0, static_cast<int>(load_err.code), String(load_err.message.c_str()),
                    String(load_err.context.c_str()));
        return static_cast<int>(load_err.code);
    }

    // Set up callbacks that queue events for main-thread delivery
    godot_llama::RequestCallbacks cbs;
    cbs.on_token = [this](const godot_llama::TokenEvent &ev) {
        std::lock_guard lock(event_mutex_);
        token_events_.push_back({ev.request_id, String(ev.text.c_str()), ev.token_id});
    };
    cbs.on_complete = [this](const godot_llama::GenerateResult &res) {
        Dictionary stats;
        stats["tokens_generated"] = res.tokens_generated;
        stats["time_ms"] = res.time_ms;
        stats["tokens_per_second"] = res.tokens_per_second;

        std::lock_guard lock(event_mutex_);
        complete_events_.push_back({res.request_id, String(res.full_text.c_str()), stats});
    };
    cbs.on_error = [this](const godot_llama::ErrorEvent &ev) {
        std::lock_guard lock(event_mutex_);
        error_events_.push_back(
                {ev.request_id, static_cast<int>(ev.code), String(ev.message.c_str()), String(ev.details.c_str())});
    };
    cbs.on_cancelled = [this](godot_llama::RequestId id) {
        std::lock_guard lock(event_mutex_);
        cancel_events_.push_back({id});
    };

    auto start_err = worker_->start(std::move(model), internal_config, std::move(cbs));
    if (start_err) {
        UtilityFunctions::push_error(String("LlamaSession: ") + start_err.message.c_str());
        return static_cast<int>(start_err.code);
    }

    is_open_ = true;
    emit_signal("opened");
    return static_cast<int>(godot_llama::ErrorCode::Ok);
}

void LlamaSession::close() {
    if (!is_open_) {
        return;
    }
    worker_->stop();
    is_open_ = false;
}

bool LlamaSession::is_open() const {
    return is_open_;
}

int LlamaSession::generate_async(const String &prompt, const Dictionary &options) {
    if (!is_open_) {
        UtilityFunctions::push_error("LlamaSession: not open");
        return -1;
    }

    auto utf8 = prompt.utf8();
    std::string prompt_str(utf8.get_data(), static_cast<size_t>(utf8.length()));
    auto opts = to_internal_options(options);

    return worker_->submit(std::move(prompt_str), std::move(opts));
}

void LlamaSession::cancel(int request_id) {
    if (is_open_) {
        worker_->cancel(static_cast<godot_llama::RequestId>(request_id));
    }
}

PackedInt32Array LlamaSession::tokenize(const String &text, bool add_bos, bool special) {
    PackedInt32Array result;
    if (!is_open_) {
        UtilityFunctions::push_error("LlamaSession: not open");
        return result;
    }

    auto utf8 = text.utf8();
    std::string_view sv(utf8.get_data(), static_cast<size_t>(utf8.length()));
    auto tokens = worker_->tokenize(sv, add_bos, special);

    result.resize(static_cast<int64_t>(tokens.size()));
    if (!tokens.empty()) {
        memcpy(result.ptrw(), tokens.data(), tokens.size() * sizeof(int32_t));
    }
    return result;
}

String LlamaSession::detokenize(const PackedInt32Array &tokens) {
    if (!is_open_) {
        UtilityFunctions::push_error("LlamaSession: not open");
        return {};
    }

    auto text = worker_->detokenize(tokens.ptr(), static_cast<int32_t>(tokens.size()));
    return String::utf8(text.c_str(), static_cast<int>(text.size()));
}

PackedFloat32Array LlamaSession::embed(const String &text) {
    PackedFloat32Array result;
    if (!is_open_) {
        UtilityFunctions::push_error("LlamaSession: not open");
        return result;
    }

    auto utf8 = text.utf8();
    std::string_view sv(utf8.get_data(), static_cast<size_t>(utf8.length()));

    std::vector<float> embd;
    auto err = worker_->embed(sv, embd);
    if (err) {
        UtilityFunctions::push_error(String("LlamaSession embed: ") + err.message.c_str());
        return result;
    }

    result.resize(static_cast<int64_t>(embd.size()));
    if (!embd.empty()) {
        memcpy(result.ptrw(), embd.data(), embd.size() * sizeof(float));
    }
    return result;
}

void LlamaSession::poll() {
    // Move events out of the lock to emit signals without holding the mutex
    std::vector<QueuedTokenEvent> tokens;
    std::vector<QueuedCompleteEvent> completes;
    std::vector<QueuedErrorEvent> errors;
    std::vector<QueuedCancelEvent> cancels;

    {
        std::lock_guard lock(event_mutex_);
        tokens.swap(token_events_);
        completes.swap(complete_events_);
        errors.swap(error_events_);
        cancels.swap(cancel_events_);
    }

    for (const auto &ev : tokens) {
        emit_signal("token_emitted", ev.request_id, ev.text, ev.token_id);
    }
    for (const auto &ev : completes) {
        emit_signal("completed", ev.request_id, ev.text, ev.stats);
    }
    for (const auto &ev : errors) {
        emit_signal("failed", ev.request_id, ev.error_code, ev.message, ev.details);
    }
    for (const auto &ev : cancels) {
        emit_signal("cancelled", ev.request_id);
    }
}

godot_llama::ModelConfig LlamaSession::to_internal_config(const Ref<Resource> &config) const {
    godot_llama::ModelConfig c;

    auto mc = Object::cast_to<LlamaModelConfig>(config.ptr());
    if (!mc) {
        return c;
    }

    auto path_utf8 = mc->get_model_path().utf8();
    c.model_path = std::string(path_utf8.get_data(), static_cast<size_t>(path_utf8.length()));
    c.n_ctx = mc->get_n_ctx();
    c.n_threads = mc->get_n_threads();
    c.n_batch = mc->get_n_batch();
    c.n_gpu_layers = mc->get_n_gpu_layers();
    c.seed = static_cast<uint32_t>(mc->get_seed());
    c.use_mmap = mc->get_use_mmap();
    c.use_mlock = mc->get_use_mlock();
    c.embeddings_enabled = mc->get_embeddings_enabled();

    auto tmpl_utf8 = mc->get_chat_template_override().utf8();
    c.chat_template_override = std::string(tmpl_utf8.get_data(), static_cast<size_t>(tmpl_utf8.length()));

    return c;
}

godot_llama::GenerateOptions LlamaSession::to_internal_options(const Dictionary &options) const {
    godot_llama::GenerateOptions opts;

    if (options.has("max_tokens")) {
        opts.max_tokens = static_cast<int32_t>(options["max_tokens"]);
    }
    if (options.has("temperature")) {
        opts.temperature = static_cast<float>(static_cast<double>(options["temperature"]));
    }
    if (options.has("top_p")) {
        opts.top_p = static_cast<float>(static_cast<double>(options["top_p"]));
    }
    if (options.has("top_k")) {
        opts.top_k = static_cast<int32_t>(options["top_k"]);
    }
    if (options.has("min_p")) {
        opts.min_p = static_cast<float>(static_cast<double>(options["min_p"]));
    }
    if (options.has("repeat_penalty")) {
        opts.repeat_penalty = static_cast<float>(static_cast<double>(options["repeat_penalty"]));
    }
    if (options.has("seed_override")) {
        opts.seed_override = static_cast<uint32_t>(static_cast<int32_t>(options["seed_override"]));
    }
    if (options.has("stop")) {
        Array stop_arr = options["stop"];
        for (int i = 0; i < stop_arr.size(); ++i) {
            auto s_utf8 = String(stop_arr[i]).utf8();
            opts.stop.emplace_back(s_utf8.get_data(), static_cast<size_t>(s_utf8.length()));
        }
    }

    return opts;
}

} // namespace godot
