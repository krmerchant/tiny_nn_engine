#pragma once
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include "model/model.h"
#include "runtime/profiler.h"
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

    virtual const Model& model_ref() const = 0;
};
inline IExecutor::~IExecutor() = default;

// ---------------------------------------------------------------------------
// GPUExecutor — GPU executor backed by a CUDA stream
// ---------------------------------------------------------------------------

class GPUExecutor : public IExecutor {
public:
    class Builder {
    public:
        Builder& model(const Model& m) { model_ = std::cref(m); return *this; }
        Builder& precision(Precision p) { precision_ = p; return *this; }
        Builder& enable_profiling(bool v) { profiling_ = v; return *this; }

        std::unique_ptr<GPUExecutor> build() {
            if (!model_.has_value())
                throw std::logic_error("GPUExecutor::Builder: model not set");
            auto exec = std::unique_ptr<GPUExecutor>(new GPUExecutor());
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

    ~GPUExecutor();

    std::unordered_map<std::string, Tensor>
    run(std::unordered_map<std::string, Tensor> inputs) override;

    const Model& model_ref() const override { return *model_; }

    // Print per-phase timing breakdown averaged over num_samples runs.
    void print_profile(int num_samples) const;

private:
    friend class Builder;
    GPUExecutor() = default;

    const Model*   model_  = nullptr;
    Precision      precision_ = Precision::FP32;
    bool           enable_profiling_ = false;
    cudaStream_t   stream_ = nullptr;

    std::unordered_map<std::string, Tensor> value_map_;

    // Profiling state
    Profiler profiler_;
    struct PhaseStats { double wall_ms = 0; double gpu_ms = 0; int count = 0; };
    mutable std::unordered_map<std::string, PhaseStats> phase_stats_;
    std::vector<std::string> phase_order_;  // insertion order for printing
};

// ---------------------------------------------------------------------------
// CPUExecutor — CPU-only executor (no CUDA required)
// ---------------------------------------------------------------------------

class CPUExecutor : public IExecutor {
public:
    class Builder {
    public:
        Builder& model(const Model& m) { model_ = std::cref(m); return *this; }

        std::unique_ptr<CPUExecutor> build() {
            if (!model_.has_value())
                throw std::logic_error("CPUExecutor::Builder: model not set");
            auto exec = std::unique_ptr<CPUExecutor>(new CPUExecutor());
            exec->model_ = &model_->get();
            for (const auto& [name, tensor] : exec->model_->graph().initializers)
                exec->value_map_[name] = tensor.clone();  // stays on CPU
            return exec;
        }

    private:
        std::optional<std::reference_wrapper<const Model>> model_;
    };

    std::unordered_map<std::string, Tensor>
    run(std::unordered_map<std::string, Tensor> inputs) override;

    const Model& model_ref() const override { return *model_; }

private:
    friend class Builder;
    CPUExecutor() = default;

    const Model* model_ = nullptr;
    std::unordered_map<std::string, Tensor> value_map_;
};

}  // namespace tinyinfer
