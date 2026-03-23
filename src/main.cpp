#include "data/mnist_loader.h"
#include "eval/evaluator.h"
#include "model/model.h"
#include "runtime/executor.h"
#include <chrono>
#include <getopt.h>
#include <iostream>

static void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " -m <model.onnx> -i <images> -l <labels> [-b <batch>]\n"
            << "\n"
            << "  -m, --model   Path to ONNX model file\n"
            << "  -i, --images  Path to IDX images binary\n"
            << "  -l, --labels  Path to IDX labels binary\n"
            << "  -b, --batch   GPU batch size (default: 64)\n"
            << "  -h, --help    Print this message\n";
}

int main(int argc, char *argv[]) {
  const char* model_path  = nullptr;
  const char* images_path = nullptr;
  const char* labels_path = nullptr;
  size_t      batch_size  = 64;

  static const option long_opts[] = {
    {"model",  required_argument, nullptr, 'm'},
    {"images", required_argument, nullptr, 'i'},
    {"labels", required_argument, nullptr, 'l'},
    {"batch",  required_argument, nullptr, 'b'},
    {"help",   no_argument,       nullptr, 'h'},
    {nullptr,  0,                 nullptr,  0 }
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:l:b:h", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'm': model_path  = optarg; break;
      case 'i': images_path = optarg; break;
      case 'l': labels_path = optarg; break;
      case 'b': batch_size  = static_cast<size_t>(std::stoul(optarg)); break;
      case 'h': print_usage(argv[0]); return 0;
      default:  print_usage(argv[0]); return 1;
    }
  }

  if (!model_path || !images_path || !labels_path) {
    std::cerr << "Error: -m, -i, and -l are required.\n\n";
    print_usage(argv[0]);
    return 1;
  }

  auto model = tinyinfer::Model::load(model_path);
  model.print_graph();

  tinyinfer::MNISTDataset dataset(images_path, labels_path);
  auto gpu_evaluator = tinyinfer::Evaluator::Builder().batch_size(batch_size).build();
  auto cpu_evaluator = tinyinfer::Evaluator::Builder().batch_size(1).build();

  // --- GPU ---
  std::cout << "\n=== GPUExecutor (batch=" << batch_size << ") ===\n";
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
  gpu_exec->print_profile(static_cast<int>(dataset.size() / batch_size));

  // --- CPU ---
  std::cout << "\n=== CPUExecutor (batch=1) ===\n";
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
