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

EvalResult Evaluator::evaluate(Executor &executor,
                               const IDataset<MNISTSample> &dataset) const {
  EvalResult result;

  const Graph &g = executor.model_ref().graph();
  const std::string &input_name  = g.input_names[0];
  const std::string &output_name = g.output_names[0];

  // 10×10 confusion matrix on CPU, zero-initialized
  result.confusion = Tensor({10, 10});
  std::memset(result.confusion.data_ptr(), 0, 100 * sizeof(float));

  for (size_t i = 0; i < dataset.size(); ++i) {
    const MNISTSample &sample = dataset[i];

    // Run inference — reshape flat image to [1, n] to add batch dim
    Tensor img = sample.image.clone();
    const auto& s = img.shape();
    if (s.size() == 1) img.reshape_({1, s[0]});
    std::unordered_map<std::string, Tensor> inputs;
    inputs[input_name] = std::move(img);
    auto outputs = executor.run(std::move(inputs));

    // Argmax over 10 classes
    Tensor &out = outputs.at(output_name);
    out.cpu();
    const float *ptr = out.data_ptr();
    int predicted = 0;
    for (int c = 1; c < 10; ++c)
      if (ptr[c] > ptr[predicted]) predicted = c;

    if (predicted == sample.label) ++result.correct;
    result.confusion.data_ptr()[sample.label * 10 + predicted] += 1.f;
  }

  result.total    = static_cast<int>(dataset.size());
  result.accuracy = static_cast<float>(result.correct) / result.total;
  return result;
}

} // namespace tinyinfer
