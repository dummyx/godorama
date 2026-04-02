#include "godot_llama/rag/factories.hpp"

namespace godot_llama::rag {
namespace {

class NoopReranker final : public Reranker {
public:
    explicit NoopReranker(const char *status) : status_(status ? status : "disabled") {}

    [[nodiscard]] bool is_available() const noexcept override { return false; }

    [[nodiscard]] Error rerank(std::string_view, std::vector<RetrievalHit> &, const CancelCheck &) const override {
        return Error::make_ok();
    }

    [[nodiscard]] const char *status_name() const noexcept override { return status_; }

private:
    const char *status_;
};

} // namespace

std::unique_ptr<Reranker> make_noop_reranker(const char *status_name) {
    return std::make_unique<NoopReranker>(status_name);
}

} // namespace godot_llama::rag
