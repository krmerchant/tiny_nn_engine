#include "data/mnist_loader.h"
#include "eval/evaluator.h"
#include "model/model.h"
#include "runtime/executor.h"
#include <chrono>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model.onnx> <images_path> <labels_path>\n";
    return 1;
  }

  auto model = tinyinfer::Model::load(argv[1]);
  model.print_graph();

  tinyinfer::MNISTDataset dataset(argv[2], argv[3]);
  auto gpu_evaluator = tinyinfer::Evaluator::Builder().batch_size(64).build();
  auto cpu_evaluator = tinyinfer::Evaluator::Builder().batch_size(1).build();

  // --- GPU ---
  std::cout << "\n=== GPUExecutor ===\n";
  auto gpu_exec = tinyinfer::GPUExecutor::Builder()
                      .model(model)
                      .enable_profiling(true)
                      .build();
  auto t0 = std::chrono::high_resolution_clock::now();
  auto gpu_result = gpu_evaluator.evaluate(*gpu_exec, dataset);
  auto t1 = std::chrono::high_resolution_clock::now();
  gpu_result.print();
  double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "Total time: " << gpu_ms << " ms"
            << "  (" << gpu_ms / dataset.size() << " ms/sample)\n";
  gpu_exec->print_profile(static_cast<int>(dataset.size() / 64));

  // --- CPU ---
  std::cout << "\n=== CPUExecutor ===\n";
  auto cpu_exec = tinyinfer::CPUExecutor::Builder().model(model).build();
  auto t2 = std::chrono::high_resolution_clock::now();
  auto cpu_result = cpu_evaluator.evaluate(*cpu_exec, dataset);
  auto t3 = std::chrono::high_resolution_clock::now();
  cpu_result.print();
  double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  std::cout << "Total time: " << cpu_ms << " ms"
            << "  (" << cpu_ms / dataset.size() << " ms/sample)\n";

  // --- Summary ---
  std::cout << "\n=== Speedup ===\n";
  std::cout << "GPU " << cpu_ms / gpu_ms << "x faster than CPU\n";

  return 0;
}
