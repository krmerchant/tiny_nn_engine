#include "onnx_parser.h"
#include "onnx.pb.h"

#include <fstream>
#include <stdexcept>
#include <cstring>

namespace tinyinfer {
namespace internal {

Graph parse_onnx(const std::string& path) {
    // Read file into string (protobuf LITE runtime requires ParseFromString)
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("parse_onnx: cannot open file: " + path);
    }
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());

    onnx::ModelProto model_proto;
    if (!model_proto.ParseFromString(content)) {
        throw std::runtime_error("parse_onnx: failed to parse protobuf from: " + path);
    }

    const onnx::GraphProto& g = model_proto.graph();
    Graph graph;

    // --- Initializers (weights) ---
    for (const auto& tp : g.initializer()) {
        if (tp.data_type() != 1) {  // 1 == FLOAT
            throw std::runtime_error(
                "parse_onnx: initializer '" + tp.name() +
                "' has unsupported data type " + std::to_string(tp.data_type()) +
                " (only FLOAT / data_type=1 is supported)");
        }
        std::vector<int64_t> shape(tp.dims().begin(), tp.dims().end());
   
        //get number of elements in flatvector
        int64_t n = 1; 
        for (auto d : shape) n *= d;

        std::vector<float> data(n);
        if (!tp.raw_data().empty()) {
            std::memcpy(data.data(), tp.raw_data().data(), n * sizeof(float));
        } else {
            std::copy(tp.float_data().begin(), tp.float_data().end(), data.begin());
        }
        graph.initializers[tp.name()] = Tensor(data, shape);
    }

    // --- Graph inputs (skip weight-only inputs that are initializers) ---
    for (const auto& vi : g.input()) {
        if (graph.initializers.find(vi.name()) == graph.initializers.end()) {
            graph.input_names.push_back(vi.name());
        }
    }

    // --- Graph outputs ---
    for (const auto& vi : g.output()) {
        graph.output_names.push_back(vi.name());
    }

    // --- Nodes (already topologically sorted per ONNX spec) ---
    for (const auto& np : g.node()) {
        Node node;
        node.name = np.name();
        //@todo replace this with unordered map
        const std::string& op = np.op_type();
        if      (op == "Gemm")    node.op_type = OpType::Gemm;
        else if (op == "Relu")    node.op_type = OpType::Relu;
        else if (op == "Softmax") node.op_type = OpType::Softmax;
        else if (op == "Flatten") node.op_type = OpType::Flatten;
        else                      node.op_type = OpType::Unknown;

        node.inputs.assign(np.input().begin(), np.input().end());
        node.outputs.assign(np.output().begin(), np.output().end());

        for (const auto& attr : np.attribute()) {
            if (attr.type() == onnx::AttributeProto::FLOAT) {
                node.float_attrs[attr.name()] = attr.f();
            } else if (attr.type() == onnx::AttributeProto::INT) {
                node.int_attrs[attr.name()] = attr.i();
            }
        }

        graph.nodes.push_back(std::move(node));
    }

    return graph;
}

}  // namespace internal
}  // namespace tinyinfer
