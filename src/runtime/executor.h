#pragma once
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include "model/model.h"
#include "tensor/tensor.h"

namespace tinyinfer {

enum class Precision { FP32 };

struct ExecutorConfig {
    Precision precision = Precision::FP32;
    bool enable_profiling = false;
};

class Executor {
public:
    Executor(const Model& model, ExecutorConfig cfg = {});
    ~Executor();

    // Run inference; returns map from output name → tensor
    std::unordered_map<std::string, Tensor>
    run(const std::unordered_map<std::string, Tensor>& inputs);

    const Model& model_ref() const { return model_; }

private:
    const Model& model_;
    ExecutorConfig cfg_;
    cudaStream_t stream_ = nullptr;

    // Persistent intermediate activation map
    std::unordered_map<std::string, Tensor> value_map_;
};

}  // namespace tinyinfer
