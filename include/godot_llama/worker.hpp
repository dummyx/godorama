#pragma once

#include "godot_llama/llama_context_handle.hpp"
#include "godot_llama/llama_model_handle.hpp"
#include "godot_llama/llama_params.hpp"
#include "godot_llama/request.hpp"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace godot_llama {

// InferenceWorker runs generation requests on a background thread.
// Thread-safe: submit/cancel can be called from any thread.
// Results are delivered via RequestCallbacks (called from the worker thread).
class InferenceWorker {
public:
    InferenceWorker() noexcept = default;
    ~InferenceWorker();

    InferenceWorker(const InferenceWorker &) = delete;
    InferenceWorker &operator=(const InferenceWorker &) = delete;

    [[nodiscard]] Error start(std::shared_ptr<LlamaModelHandle> model, const ModelConfig &config,
                              RequestCallbacks callbacks);

    void stop() noexcept;
    [[nodiscard]] bool is_running() const noexcept;

    [[nodiscard]] RequestId submit(std::string prompt, GenerateOptions options);
    [[nodiscard]] RequestId submit_with_id(RequestId request_id, std::string prompt, GenerateOptions options);
    void cancel(RequestId id) noexcept;
    [[nodiscard]] Error apply_chat_template(const std::vector<std::pair<std::string, std::string>> &messages,
                                            bool add_assistant_turn, std::string &out_prompt) const;

    // Tokenize/detokenize are synchronous and safe to call from any thread.
    [[nodiscard]] std::vector<int32_t> tokenize(std::string_view text, bool add_bos, bool special) const;
    [[nodiscard]] std::string detokenize(const int32_t *tokens, int32_t n_tokens) const;

    // Embedding (synchronous, blocking — use from worker or test code only)
    [[nodiscard]] Error embed(std::string_view text, std::vector<float> &out);

private:
    void run();
    void process_request(GenerateRequest &req);

    std::shared_ptr<LlamaModelHandle> model_;
    LlamaContextHandle context_;
    RequestCallbacks callbacks_;
    ModelConfig config_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<GenerateRequest>> queue_;
    std::shared_ptr<GenerateRequest> active_request_;
    std::jthread thread_;
    std::atomic<bool> running_{false};
    std::atomic<RequestId> next_id_{1};
};

} // namespace godot_llama
