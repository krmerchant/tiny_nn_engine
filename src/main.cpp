#include "data/mnist_loader.h"
#include "model/model.h"
#include "runtime/executor.h"
#include "tensor/tensor.h"
#include <iostream>
#include <memory>
#include <unordered_map>
int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model.onnx> <data-path> <labels_path> " << std::endl;
    return 1;
  }

  auto model = tinyinfer::Model::load(argv[1]);
  model.print_graph();

  auto exec = tinyinfer::Executor::Builder()
                  .model(model)
                  .enable_profiling(false)
                  .build();

  std::unique_ptr dataset =
      std::make_unique<tinyinfer::MNISTDataset>(argv[2], argv[3]);

  const auto &X = (*dataset)[0];
  X.image.to_matlab("test.m");
  return 0;
}
