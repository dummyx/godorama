#include "llama_session.hpp"

#include "godot_llama/llama_model_handle.hpp"
#include "llama_lora_adapter_config.hpp"
#include "llama_model_config.hpp"
#include "llama_multimodal_config.hpp"

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
    ClassDB::bind_method(D_METHOD("is_opening"), &LlamaSession::is_opening);
    ClassDB::bind_method(D_METHOD("generate_async", "prompt", "options"), &LlamaSession::generate_async,
                         DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("generate_messages_async", "messages", "options", "add_assistant_turn"),
                         &LlamaSession::generate_messages_async, DEFVAL(Dictionary()), DEFVAL(true));
    ClassDB::bind_method(D_METHOD("generate_multimodal_async", "prompt", "media_inputs", "options"),
                         &LlamaSession::generate_multimodal_async, DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("generate_multimodal_messages_async", "messages", "media_inputs", "options",
                                  "add_assistant_turn"),
                         &LlamaSession::generate_multimodal_messages_async, DEFVAL(Dictionary()), DEFVAL(true));
    ClassDB::bind_method(D_METHOD("cancel", "request_id"), &LlamaSession::cancel);
    ClassDB::bind_method(D_METHOD("tokenize", "text", "add_bos", "special"), &LlamaSession::tokenize, DEFVAL(false),
                         DEFVAL(false));
    ClassDB::bind_method(D_METHOD("detokenize", "tokens"), &LlamaSession::detokenize);
    ClassDB::bind_method(D_METHOD("embed", "text"), &LlamaSession::embed);
    ClassDB::bind_method(D_METHOD("get_lora_adapter_count"), &LlamaSession::get_lora_adapter_count);
    ClassDB::bind_method(D_METHOD("supports_image_input"), &LlamaSession::supports_image_input);
    ClassDB::bind_method(D_METHOD("supports_audio_input"), &LlamaSession::supports_audio_input);
    ClassDB::bind_method(D_METHOD("get_audio_input_sample_rate_hz"), &LlamaSession::get_audio_input_sample_rate_hz);
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
    if (is_open_.load(std::memory_order_acquire) || is_opening_.load(std::memory_order_acquire) ||
        !open_thread_finished_.load(std::memory_order_acquire)) {
        UtilityFunctions::push_error("LlamaSession: already open, call close() first");
        return static_cast<int>(godot_llama::ErrorCode::AlreadyOpen);
    }

    if (config.is_null()) {
        UtilityFunctions::push_error("LlamaSession: config is null");
        return static_cast<int>(godot_llama::ErrorCode::InvalidParameter);
    }

    auto internal_config = to_internal_config(config);
    is_opening_.store(true, std::memory_order_release);
    open_thread_finished_.store(false, std::memory_order_release);
    const uint64_t open_generation = open_generation_.fetch_add(1, std::memory_order_acq_rel) + 1;

    open_thread_ = std::jthread([this, internal_config, open_generation](std::stop_token stop_token) {
        std::shared_ptr<godot_llama::LlamaModelHandle> model;
        const auto load_err = godot_llama::LlamaModelHandle::load(internal_config, model);

        if (load_err) {
            if (!is_stale_open_generation(open_generation) && !stop_token.stop_requested()) {
                enqueue_open_failed_event(load_err);
            }
            finalize_open_thread();
            return;
        }

        if (is_stale_open_generation(open_generation) || stop_token.stop_requested()) {
            finalize_open_thread();
            return;
        }

        godot_llama::RequestCallbacks cbs;
        cbs.on_token = [this](const godot_llama::TokenEvent &ev) {
            std::lock_guard lock(event_mutex_);
            token_events_.push_back(
                    {ev.request_id, String::utf8(ev.text.c_str(), static_cast<int>(ev.text.size())), ev.token_id});
        };
        cbs.on_complete = [this](const godot_llama::GenerateResult &res) {
            Dictionary stats;
            stats["tokens_generated"] = res.tokens_generated;
            stats["time_ms"] = res.time_ms;
            stats["tokens_per_second"] = res.tokens_per_second;

            std::lock_guard lock(event_mutex_);
            complete_events_.push_back(
                    {res.request_id, String::utf8(res.full_text.c_str(), static_cast<int>(res.full_text.size())),
                     stats});
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

        {
            std::lock_guard lock(state_mutex_);
            if (is_stale_open_generation(open_generation) || stop_token.stop_requested()) {
                finalize_open_thread();
                return;
            }

            const auto start_err = worker_->start(std::move(model), internal_config, std::move(cbs));
            if (start_err) {
                enqueue_open_failed_event(start_err);
                finalize_open_thread();
                return;
            }

            is_open_.store(true, std::memory_order_release);
        }

        if (is_stale_open_generation(open_generation) || stop_token.stop_requested()) {
            std::lock_guard lock(state_mutex_);
            if (is_open_.exchange(false, std::memory_order_acq_rel)) {
                worker_->stop();
            }
            finalize_open_thread();
            return;
        }

        enqueue_opened_event();
        finalize_open_thread();
    });

    return static_cast<int>(godot_llama::ErrorCode::Ok);
}

void LlamaSession::close() {
    open_generation_.fetch_add(1, std::memory_order_acq_rel);
    is_opening_.store(false, std::memory_order_release);

    if (open_thread_.joinable()) {
        open_thread_.request_stop();
        if (open_thread_finished_.load(std::memory_order_acquire)) {
            open_thread_.join();
        }
    }

    std::lock_guard lock(state_mutex_);
    if (!is_open_.exchange(false, std::memory_order_acq_rel)) {
        return;
    }

    worker_->stop();
}

bool LlamaSession::is_open() const {
    return is_open_.load(std::memory_order_acquire);
}

bool LlamaSession::is_opening() const {
    return is_opening_.load(std::memory_order_acquire);
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

int LlamaSession::generate_messages_async(const Array &messages, const Dictionary &options, bool add_assistant_turn) {
    if (!is_open_) {
        UtilityFunctions::push_error("LlamaSession: not open");
        return -1;
    }

    const auto internal_messages = to_internal_messages(messages);
    if (internal_messages.empty()) {
        UtilityFunctions::push_error("LlamaSession: messages must contain at least one valid {role, content} entry");
        return -static_cast<int>(godot_llama::ErrorCode::InvalidParameter);
    }

    std::string prompt;
    std::vector<std::string> template_stops;
    const auto template_err = worker_->apply_chat_template(internal_messages, add_assistant_turn, prompt, template_stops);
    if (template_err) {
        UtilityFunctions::push_error(String("LlamaSession generate_messages_async: ") + template_err.message.c_str());
        return -static_cast<int>(template_err.code);
    }

    auto opts = to_internal_options(options);
    for (auto &s : template_stops) {
        opts.stop.push_back(std::move(s));
    }
    return worker_->submit(std::move(prompt), std::move(opts), /* prompt_has_special_tokens */ true);
}

int LlamaSession::generate_multimodal_async(const String &prompt, const Array &media_inputs,
                                            const Dictionary &options) {
    if (!is_open_) {
        UtilityFunctions::push_error("LlamaSession: not open");
        return -1;
    }

    const auto internal_media = to_internal_media_inputs(media_inputs);
    if (internal_media.empty()) {
        UtilityFunctions::push_error("LlamaSession: media_inputs must contain at least one valid {path, type} entry");
        return -static_cast<int>(godot_llama::ErrorCode::InvalidParameter);
    }

    auto utf8 = prompt.utf8();
    std::string prompt_str(utf8.get_data(), static_cast<size_t>(utf8.length()));
    auto opts = to_internal_options(options);

    return worker_->submit_multimodal(std::move(prompt_str), std::move(internal_media), std::move(opts));
}

int LlamaSession::generate_multimodal_messages_async(const Array &messages, const Array &media_inputs,
                                                     const Dictionary &options, bool add_assistant_turn) {
    if (!is_open_) {
        UtilityFunctions::push_error("LlamaSession: not open");
        return -1;
    }

    const auto internal_messages = to_internal_messages(messages);
    if (internal_messages.empty()) {
        UtilityFunctions::push_error("LlamaSession: messages must contain at least one valid {role, content} entry");
        return -static_cast<int>(godot_llama::ErrorCode::InvalidParameter);
    }

    const auto internal_media = to_internal_media_inputs(media_inputs);
    if (internal_media.empty()) {
        UtilityFunctions::push_error("LlamaSession: media_inputs must contain at least one valid {path, type} entry");
        return -static_cast<int>(godot_llama::ErrorCode::InvalidParameter);
    }

    std::string prompt;
    std::vector<std::string> template_stops;
    const auto template_err = worker_->apply_chat_template(internal_messages, add_assistant_turn, prompt, template_stops);
    if (template_err) {
        UtilityFunctions::push_error(
                String("LlamaSession generate_multimodal_messages_async: ") + template_err.message.c_str());
        return -static_cast<int>(template_err.code);
    }

    auto opts = to_internal_options(options);
    for (auto &s : template_stops) {
        opts.stop.push_back(std::move(s));
    }
    return worker_->submit_multimodal(std::move(prompt), std::move(internal_media), std::move(opts),
                                      /* prompt_has_special_tokens */ true);
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

int LlamaSession::get_lora_adapter_count() const {
    std::lock_guard lock(state_mutex_);
    return worker_ ? static_cast<int>(worker_->lora_adapter_count()) : 0;
}

bool LlamaSession::supports_image_input() const {
    std::lock_guard lock(state_mutex_);
    return worker_ ? worker_->supports_image_input() : false;
}

bool LlamaSession::supports_audio_input() const {
    std::lock_guard lock(state_mutex_);
    return worker_ ? worker_->supports_audio_input() : false;
}

int LlamaSession::get_audio_input_sample_rate_hz() const {
    std::lock_guard lock(state_mutex_);
    return worker_ ? worker_->audio_input_sample_rate_hz() : -1;
}

void LlamaSession::poll() {
    // Move events out of the lock to emit signals without holding the mutex
    std::vector<QueuedOpenedEvent> opened;
    std::vector<QueuedTokenEvent> tokens;
    std::vector<QueuedCompleteEvent> completes;
    std::vector<QueuedErrorEvent> errors;
    std::vector<QueuedCancelEvent> cancels;

    {
        std::lock_guard lock(event_mutex_);
        opened.swap(opened_events_);
        tokens.swap(token_events_);
        completes.swap(complete_events_);
        errors.swap(error_events_);
        cancels.swap(cancel_events_);
    }

    for (size_t i = 0; i < opened.size(); ++i) {
        emit_signal("opened");
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

    if (open_thread_.joinable() && open_thread_finished_.load(std::memory_order_acquire)) {
        open_thread_.join();
    }
}

void LlamaSession::enqueue_opened_event() {
    std::lock_guard lock(event_mutex_);
    opened_events_.push_back({});
}

void LlamaSession::enqueue_open_failed_event(const godot_llama::Error &error) {
    std::lock_guard lock(event_mutex_);
    error_events_.push_back({0, static_cast<int>(error.code), String(error.message.c_str()), String(error.context.c_str())});
}

void LlamaSession::finalize_open_thread() noexcept {
    is_opening_.store(false, std::memory_order_release);
    open_thread_finished_.store(true, std::memory_order_release);
}

bool LlamaSession::is_stale_open_generation(uint64_t generation) const noexcept {
    return generation != open_generation_.load(std::memory_order_acquire);
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
    c.disable_thinking = mc->get_disable_thinking();
    c.flash_attn_type = mc->get_flash_attn_type();
    c.type_k = mc->get_type_k();
    c.type_v = mc->get_type_v();

    auto tmpl_utf8 = mc->get_chat_template_override().utf8();
    c.chat_template_override = std::string(tmpl_utf8.get_data(), static_cast<size_t>(tmpl_utf8.length()));

    const Array lora_adapters = mc->get_lora_adapters();
    for (int64_t index = 0; index < lora_adapters.size(); ++index) {
        const Variant &entry = lora_adapters[index];
        if (entry.get_type() != Variant::OBJECT) {
            continue;
        }

        Object *entry_object = entry;
        const auto *adapter = Object::cast_to<LlamaLoraAdapterConfig>(entry_object);
        if (adapter == nullptr) {
            continue;
        }

        auto path_utf8 = adapter->get_adapter_path().utf8();
        godot_llama::LoraAdapterConfig adapter_config;
        adapter_config.path = std::string(path_utf8.get_data(), static_cast<size_t>(path_utf8.length()));
        adapter_config.scale = static_cast<float>(adapter->get_scale());
        c.lora_adapters.push_back(std::move(adapter_config));
    }

    const Ref<LlamaMultimodalConfig> multimodal = mc->get_multimodal_config();
    if (!multimodal.is_null()) {
        godot_llama::MultimodalConfig multimodal_config;
        auto mmproj_utf8 = multimodal->get_mmproj_path().utf8();
        multimodal_config.mmproj_path =
                std::string(mmproj_utf8.get_data(), static_cast<size_t>(mmproj_utf8.length()));
        auto marker_utf8 = multimodal->get_media_marker().utf8();
        multimodal_config.media_marker =
                std::string(marker_utf8.get_data(), static_cast<size_t>(marker_utf8.length()));
        multimodal_config.use_gpu = multimodal->get_use_gpu();
        multimodal_config.print_timings = multimodal->get_print_timings();
        multimodal_config.n_threads = multimodal->get_n_threads();
        multimodal_config.image_min_tokens = multimodal->get_image_min_tokens();
        multimodal_config.image_max_tokens = multimodal->get_image_max_tokens();
        c.multimodal = std::move(multimodal_config);
    }

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

std::vector<std::pair<std::string, std::string>> LlamaSession::to_internal_messages(const Array &messages) const {
    std::vector<std::pair<std::string, std::string>> internal_messages;
    internal_messages.reserve(static_cast<size_t>(messages.size()));

    for (int i = 0; i < messages.size(); ++i) {
        const Dictionary message = messages[i];
        if (!message.has("role") || !message.has("content")) {
            continue;
        }

        const String role = String(message["role"]).strip_edges();
        const String content = String(message["content"]).strip_edges();
        if (role.is_empty() || content.is_empty()) {
            continue;
        }

        const auto role_utf8 = role.utf8();
        const auto content_utf8 = content.utf8();
        internal_messages.emplace_back(
                std::string(role_utf8.get_data(), static_cast<size_t>(role_utf8.length())),
                std::string(content_utf8.get_data(), static_cast<size_t>(content_utf8.length())));
    }

    return internal_messages;
}

std::vector<godot_llama::MultimodalInput> LlamaSession::to_internal_media_inputs(const Array &media_inputs) const {
    std::vector<godot_llama::MultimodalInput> result;
    result.reserve(static_cast<size_t>(media_inputs.size()));

    for (int64_t i = 0; i < media_inputs.size(); ++i) {
        const Variant &entry = media_inputs[i];
        if (entry.get_type() != Variant::DICTIONARY) {
            continue;
        }

        const Dictionary media = entry;
        if (!media.has("path") && !media.has("data")) {
            continue;
        }

        godot_llama::MultimodalInput input;
        if (media.has("type")) {
            const String type = String(media["type"]).strip_edges().to_lower();
            if (type == "audio" || type == "voice" || type == "speech") {
                input.type = godot_llama::MultimodalInputType::Audio;
            }
        }

        if (media.has("path")) {
            const String path = String(media["path"]).strip_edges();
            if (!path.is_empty()) {
                auto path_utf8 = path.utf8();
                input.path = std::string(path_utf8.get_data(), static_cast<size_t>(path_utf8.length()));
            }
        }

        if (media.has("data")) {
            const PackedByteArray bytes = media["data"];
            if (bytes.size() > 0) {
                input.data.resize(static_cast<size_t>(bytes.size()));
                memcpy(input.data.data(), bytes.ptr(), static_cast<size_t>(bytes.size()));
            }
        }

        if (media.has("id")) {
            const String id = String(media["id"]).strip_edges();
            if (!id.is_empty()) {
                auto id_utf8 = id.utf8();
                input.id = std::string(id_utf8.get_data(), static_cast<size_t>(id_utf8.length()));
            }
        }

        // Accept entries that have either a path or in-memory data
        if (!input.path.empty() || !input.data.empty()) {
            result.push_back(std::move(input));
        }
    }

    return result;
}

} // namespace godot
