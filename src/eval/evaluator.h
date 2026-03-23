#pragma once
#include <cstddef>
#include <vector>
#include "data/mnist_loader.h"
#include "runtime/executor.h"

namespace tinyinfer {

struct EvalConfig {
    size_t batch_size = 1;   // samples per inference call
};

struct EvalResult {
    float accuracy = 0.f;            // fraction correct
    int correct = 0;
    int total = 0;
    std::vector<std::vector<int>> confusion;  // 10×10

    void print() const;
};

class Evaluator {
public:
    // Run the full dataset through executor, collect accuracy and confusion matrix
    static EvalResult evaluate(Executor& executor,
                               const IDataset<MNISTSample>& dataset,
                               const EvalConfig& cfg = {});
};

}  // namespace tinyinfer
