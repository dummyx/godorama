#pragma once

#include "godot_llama/llama_context_handle.hpp"
#include "godot_llama/llama_model_handle.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace godot {

class LlamaEvalSession : public RefCounted {
    GDCLASS(LlamaEvalSession, RefCounted)

public:
    LlamaEvalSession() = default;
    ~LlamaEvalSession() override;

    int open(const Ref<Resource> &config);
    void close();

    bool is_open() const;
    bool is_opening() const;

    int run_prefill_async(const PackedFloat32Array &inputs_embeds, int32_t sequence_length,
                          const PackedInt32Array &position_ids = PackedInt32Array(), int32_t position_components = 1,
                          int32_t logit_start = 0, int32_t logit_count = 0, bool include_hidden_state = true,
                          bool clear_kv_cache = true);
    void cancel(int request_id);
    void poll();

protected:
    static void _bind_methods();

private:
    struct EvalRequest {
        int request_id = 0;
        std::vector<float> embeddings;
        std::vector<int32_t> positions;
        int32_t sequence_length = 0;
        int32_t position_components = 1;
        int32_t logit_start = 0;
        int32_t logit_count = 0;
        bool include_hidden_state = true;
        bool clear_kv_cache = true;
        std::atomic<bool> cancelled{false};
    };

    struct QueuedOpenedEvent {};
    struct QueuedCompleteEvent {
        int request_id = 0;
        Dictionary result;
    };
    struct QueuedErrorEvent {
        int request_id = 0;
        int error_code = 0;
        String message;
        String details;
    };
    struct QueuedCancelEvent {
        int request_id = 0;
    };

    godot_llama::ModelConfig to_internal_config(const Ref<Resource> &config) const;
    void worker_loop(std::stop_token stop_token);
    void process_request(const std::shared_ptr<EvalRequest> &request);
    void enqueue_opened_event();
    void enqueue_complete_event(int request_id, const Dictionary &result);
    void enqueue_error_event(int request_id, const godot_llama::Error &error);
    void enqueue_cancel_event(int request_id);
    void finalize_open_thread() noexcept;
    [[nodiscard]] bool is_stale_open_generation(uint64_t generation) const noexcept;

    std::mutex event_mutex_;
    std::vector<QueuedOpenedEvent> opened_events_;
    std::vector<QueuedCompleteEvent> complete_events_;
    std::vector<QueuedErrorEvent> error_events_;
    std::vector<QueuedCancelEvent> cancel_events_;

    std::mutex queue_mutex_;
    std::condition_variable_any queue_cv_;
    std::deque<std::shared_ptr<EvalRequest>> request_queue_;
    std::shared_ptr<EvalRequest> active_request_;

    mutable std::mutex state_mutex_;
    std::shared_ptr<godot_llama::LlamaModelHandle> model_;
    godot_llama::LlamaContextHandle context_;
    std::jthread open_thread_;
    std::jthread worker_thread_;
    std::atomic<bool> is_open_{false};
    std::atomic<bool> is_opening_{false};
    std::atomic<bool> open_thread_finished_{true};
    std::atomic<uint64_t> open_generation_{0};
    std::atomic<int> next_request_id_{0};
};

} // namespace godot
