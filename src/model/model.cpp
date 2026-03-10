#include "model.h"
#include "onnx_parser.h"
#include <iostream>

namespace tinyinfer {

Model Model::load(const std::string& onnx_path) {
    Graph g = internal::parse_onnx(onnx_path);
    return Model(std::move(g));
}

void Model::print_graph() const {
    auto op_type_str = [](OpType t) -> const char* {
        switch (t) {
            case OpType::Gemm:    return "Gemm";
            case OpType::Relu:    return "Relu";
            case OpType::Softmax: return "Softmax";
            case OpType::Flatten: return "Flatten";
            default:              return "Unknown";
        }
    };

    std::cout << "Graph (" << graph_.nodes.size() << " nodes):\n";
    for (const auto& node : graph_.nodes) {
        std::cout << "  [" << op_type_str(node.op_type) << "]"
                  << " name=" << node.name << "\n";
        std::cout << "    inputs:";
        for (const auto& s : node.inputs)  std::cout << " " << s;
        std::cout << "\n    outputs:";
        for (const auto& s : node.outputs) std::cout << " " << s;
        std::cout << "\n";
    }

    std::cout << "Initializers:";
    for (const auto& kv : graph_.initializers) std::cout << " " << kv.first;
    std::cout << "\n";

    std::cout << "Inputs:";
    for (const auto& s : graph_.input_names)  std::cout << " " << s;
    std::cout << "\nOutputs:";
    for (const auto& s : graph_.output_names) std::cout << " " << s;
    std::cout << "\n";
}

}  // namespace tinyinfer
