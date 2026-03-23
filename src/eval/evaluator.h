#pragma once
#include <cstddef>
#include <cstring>
#include <vector>
#include "data/mnist_loader.h"
#include "runtime/executor.h"
#include "tensor/tensor.h"

namespace tinyinfer {

struct EvalResult {
    float accuracy = 0.f;            // fraction correct
    int correct = 0;
    int total = 0;
    Tensor confusion;                // 10×10 CPU tensor, confusion[true][pred]

    void print() const;
};

class Evaluator {
public:
    class Builder {
    public:
        Builder& batch_size(size_t n) { batch_size_ = n; return *this; }
        Evaluator build() const { return Evaluator(batch_size_); }
    private:
        size_t batch_size_ = 1;
    };

    EvalResult evaluate(Executor& executor,
                        const IDataset<MNISTSample>& dataset) const;

private:
    explicit Evaluator(size_t batch_size) : batch_size_(batch_size) {}
    size_t batch_size_;
};

}  // namespace tinyinfer
