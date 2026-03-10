#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "tensor/tensor.h"

namespace tinyinfer {

enum class OpType {
    Unknown,
    Gemm,
    Relu,
    Softmax,
    Flatten,
};

struct Node {
    OpType op_type = OpType::Unknown;
    std::string name;
    std::vector<std::string> inputs;   // ONNX value names
    std::vector<std::string> outputs;  // ONNX value names

    // Op-specific attributes (e.g., transB for Gemm)
    std::unordered_map<std::string, float> float_attrs;
    std::unordered_map<std::string, int64_t> int_attrs;
};

struct Graph {
    std::vector<Node> nodes;                          // topologically sorted
    std::unordered_map<std::string, Tensor> initializers;  // weight tensors on GPU
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

}  // namespace tinyinfer
