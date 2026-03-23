#include "eval/evaluator.h"
#include "graph/graph.h"
#include <cstring>
#include <iostream>
#include <unordered_map>

namespace tinyinfer {

// ---------------------------------------------------------------------------
// EvalResult::print
// ---------------------------------------------------------------------------

void EvalResult::print() const {
  std::cout << "Accuracy: " << correct << " / " << total
            << "  (" << accuracy * 100.f << "%)\n\n";

  std::cout << "Confusion matrix (rows=true, cols=predicted):\n";
  std::cout << "     ";
  for (int c = 0; c < 10; ++c) std::cout << "  " << c << " ";
  std::cout << "\n";
  for (int r = 0; r < 10; ++r) {
    std::cout << "  " << r << "  ";
    for (int c = 0; c < 10; ++c) {
      int v = static_cast<int>(confusion.data_ptr()[r * 10 + c]);
      std::cout << "  " << v << " ";
    }
    std::cout << "\n";
  }
}

// ---------------------------------------------------------------------------
// Evaluator::evaluate
// ---------------------------------------------------------------------------

EvalResult Evaluator::evaluate(IExecutor &executor,
                               const IDataset<MNISTSample> &dataset) const {
  EvalResult result;

  const Graph &g = executor.model_ref().graph();
  const std::string &input_name  = g.input_names[0];
  const std::string &output_name = g.output_names[0];

  // 10×10 confusion matrix on CPU, zero-initialized
  result.confusion = Tensor({10, 10});
  std::memset(result.confusion.data_ptr(), 0, 100 * sizeof(float));

  const int64_t feat = dataset[0].image.shape()[0];  // 784

  for (size_t i = 0; i < dataset.size(); i += batch_size_) {
    size_t n = std::min(batch_size_, dataset.size() - i);

    // Stack n images into [n, feat]
    Tensor batch({(int64_t)n, feat});
    for (size_t j = 0; j < n; ++j)
      std::memcpy(batch.data_ptr() + j * feat,
                  dataset[i + j].image.data_ptr(),
                  feat * sizeof(float));

    std::unordered_map<std::string, Tensor> inputs;
    inputs[input_name] = std::move(batch);
    auto outputs = executor.run(std::move(inputs));

    // Argmax each row of [n, 10] output
    Tensor &out = outputs.at(output_name);
    out.cpu();
    for (size_t j = 0; j < n; ++j) {
      const float *row = out.data_ptr() + j * 10;
      int predicted = 0;
      for (int c = 1; c < 10; ++c)
        if (row[c] > row[predicted]) predicted = c;
      int label = dataset[i + j].label;
      if (predicted == label) ++result.correct;
      result.confusion.data_ptr()[label * 10 + predicted] += 1.f;
    }
  }

  result.total    = static_cast<int>(dataset.size());
  result.accuracy = static_cast<float>(result.correct) / result.total;
  return result;
}

} // namespace tinyinfer
