#pragma once
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include "model/model.h"
#include "tensor/tensor.h"

namespace tinyinfer {

enum class Precision { FP32 };

// ---------------------------------------------------------------------------
// IExecutor — abstract inference interface
// ---------------------------------------------------------------------------

class IExecutor {
public:
    virtual ~IExecutor() = 0;

    // Run one inference pass. Inputs are keyed by ONNX input name.
    // Caller moves tensors in; executor uploads to GPU as needed.
    // Returns a map from ONNX output name → result tensor.
    virtual std::unordered_map<std::string, Tensor>
    run(std::unordered_map<std::string, Tensor> inputs) = 0;
};
inline IExecutor::~IExecutor() = default;

// ---------------------------------------------------------------------------
// Executor — GPU executor backed by a CUDA stream
// ---------------------------------------------------------------------------

class Executor : public IExecutor {
public:
    class Builder {
    public:
        Builder& model(const Model& m) { model_ = std::cref(m); return *this; }
        Builder& precision(Precision p) { precision_ = p; return *this; }
        Builder& enable_profiling(bool v) { profiling_ = v; return *this; }

        std::unique_ptr<Executor> build() {
            if (!model_.has_value())
                throw std::logic_error("Executor::Builder: model not set");
            auto exec = std::unique_ptr<Executor>(new Executor());
            exec->model_            = &model_->get();
            exec->precision_        = precision_;
            exec->enable_profiling_ = profiling_;
            cudaStreamCreate(&exec->stream_);
            for (const auto& [name, tensor] : exec->model_->graph().initializers) {
                Tensor t = tensor.clone();
                t.cuda();
                exec->value_map_[name] = std::move(t);
            }
            return exec;
        }

    private:
        std::optional<std::reference_wrapper<const Model>> model_;
        Precision precision_ = Precision::FP32;
        bool      profiling_ = false;
    };

    ~Executor();

    std::unordered_map<std::string, Tensor>
    run(std::unordered_map<std::string, Tensor> inputs) override;

    const Model& model_ref() const { return *model_; }

private:
    friend class Builder;
    Executor() = default;

    const Model*   model_  = nullptr;
    Precision      precision_ = Precision::FP32;
    bool           enable_profiling_ = false;
    cudaStream_t   stream_ = nullptr;

    // Persistent intermediate activation map
    std::unordered_map<std::string, Tensor> value_map_;
};

}  // namespace tinyinfer
