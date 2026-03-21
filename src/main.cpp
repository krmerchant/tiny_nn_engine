#include "model/model.h"
#include "runtime/executor.h"
#include "tensor/tensor.h"
#include <iostream>
#include <unordered_map>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
        return 1;
    }
    auto model = tinyinfer::Model::load(argv[1]);
    model.print_graph();

    auto exec = tinyinfer::Executor::Builder()
        .model(model)
        .precision(tinyinfer::Precision::FP32)
        .enable_profiling(false)
        .build();

    // dummy all-zeros input (1×784)
    auto input = tinyinfer::Tensor({1, 784});
    input.fill(0.f);
    std::unordered_map<std::string, tinyinfer::Tensor> inputs;
    inputs.emplace("input", std::move(input));
    auto outputs = exec->run(std::move(inputs));

    std::cout << "run() completed, " << outputs.size() << " output(s)\n";
    return 0;
}
