#pragma once

#include "godot_llama/request.hpp"
#include "godot_llama/worker.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace godot {

class LlamaModelConfig;
class LlamaLoraAdapterConfig;
class LlamaMultimodalConfig;

class LlamaSession : public RefCounted {
    GDCLASS(LlamaSession, RefCounted)

public:
    LlamaSession();
    ~LlamaSession() override;

    // --- Bound methods ---

    // Non-blocking: loads model on a background thread and emits opened() or failed().
    int open(const Ref<Resource> &config);

    // Immediate: shuts down the worker and releases the model.
    void close();

    // Non-blocking query.
    bool is_open() const;
    bool is_opening() const;

    // Non-blocking: submits a generation request. Returns request_id.
    int generate_async(const String &prompt, const Dictionary &options = Dictionary());
    int generate_messages_async(const Array &messages, const Dictionary &options = Dictionary(),
                                bool add_assistant_turn = true);
    int generate_multimodal_async(const String &prompt, const Array &media_inputs,
                                  const Dictionary &options = Dictionary());
    int generate_multimodal_messages_async(const Array &messages, const Array &media_inputs,
                                           const Dictionary &options = Dictionary(),
                                           bool add_assistant_turn = true);

    // Non-blocking: cancels a pending or in-progress generation.
    void cancel(int request_id);

    // Blocking but fast: tokenize text into token IDs.
    PackedInt32Array tokenize(const String &text, bool add_bos = false, bool special = false);

    // Blocking but fast: convert token IDs back to text.
    String detokenize(const PackedInt32Array &tokens);

    // Blocking: compute embeddings for text (requires embeddings_enabled in config).
    PackedFloat32Array embed(const String &text);

    // Non-blocking queries over the current open session state.
    int get_lora_adapter_count() const;
    bool supports_image_input() const;
    bool supports_audio_input() const;
    int get_audio_input_sample_rate_hz() const;

    // Called by Godot each frame to flush queued events to signals.
    void poll();

protected:
    static void _bind_methods();

private:
    godot_llama::ModelConfig to_internal_config(const Ref<Resource> &config) const;
    godot_llama::GenerateOptions to_internal_options(const Dictionary &options) const;
    std::vector<std::pair<std::string, std::string>> to_internal_messages(const Array &messages) const;
    std::vector<godot_llama::MultimodalInput> to_internal_media_inputs(const Array &media_inputs) const;
    void enqueue_opened_event();
    void enqueue_open_failed_event(const godot_llama::Error &error);
    void finalize_open_thread() noexcept;
    [[nodiscard]] bool is_stale_open_generation(uint64_t generation) const noexcept;

    // Queued events for main-thread delivery
    struct QueuedOpenedEvent {};
    struct QueuedTokenEvent {
        int request_id;
        String text;
        int token_id;
    };
    struct QueuedCompleteEvent {
        int request_id;
        String text;
        Dictionary stats;
    };
    struct QueuedErrorEvent {
        int request_id;
        int error_code;
        String message;
        String details;
    };
    struct QueuedCancelEvent {
        int request_id;
    };

    std::mutex event_mutex_;
    std::vector<QueuedOpenedEvent> opened_events_;
    std::vector<QueuedTokenEvent> token_events_;
    std::vector<QueuedCompleteEvent> complete_events_;
    std::vector<QueuedErrorEvent> error_events_;
    std::vector<QueuedCancelEvent> cancel_events_;

    mutable std::mutex state_mutex_;
    std::unique_ptr<godot_llama::InferenceWorker> worker_;
    std::jthread open_thread_;
    std::atomic<bool> is_open_{false};
    std::atomic<bool> is_opening_{false};
    std::atomic<bool> open_thread_finished_{true};
    std::atomic<uint64_t> open_generation_{0};
};

} // namespace godot
