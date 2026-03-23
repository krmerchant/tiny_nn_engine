#include "eval/evaluator.h"
#include <iostream>

namespace tinyinfer {

// ---------------------------------------------------------------------------
// EvalResult::print
// ---------------------------------------------------------------------------

void EvalResult::print() const {
    // TODO: print accuracy and confusion matrix
}

// ---------------------------------------------------------------------------
// Evaluator::evaluate
// ---------------------------------------------------------------------------

EvalResult Evaluator::evaluate(Executor& executor,
                               const IDataset<MNISTSample>& dataset,
                               const EvalConfig& cfg) {
    EvalResult result;

    // TODO: initialize 10×10 confusion matrix

    for (size_t i = 0; i < dataset.size(); ++i) {
        const MNISTSample& sample = dataset[i];

        // TODO: copy sample.image into an input tensor map and call executor.run()

        // TODO: run argmax on the output to get predicted class

        // TODO: compare prediction to sample.label, increment result.correct

        // TODO: update confusion matrix at [sample.label][predicted]
    }

    result.total = static_cast<int>(dataset.size());
    // TODO: compute result.accuracy from result.correct / result.total

    return result;
}

}  // namespace tinyinfer
