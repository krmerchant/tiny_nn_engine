#include "data/mnist_loader.h"
#include "eval/evaluator.h"
#include "model/model.h"
#include "runtime/executor.h"
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model.onnx> <images_path> <labels_path>\n";
    return 1;
  }

  auto model = tinyinfer::Model::load(argv[1]);
  model.print_graph();

  auto exec = tinyinfer::Executor::Builder()
                  .model(model)
                  .enable_profiling(false)
                  .build();

  tinyinfer::MNISTDataset dataset(argv[2], argv[3]);

  auto evaluator = tinyinfer::Evaluator::Builder().build();
  auto result    = evaluator.evaluate(*exec, dataset);
  result.print();

  return 0;
}
