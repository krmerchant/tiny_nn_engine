#include <fstream>
#include <iostream>
#include <string>

#include "onnx.pb.h"

static std::string shape_str(const onnx::TypeProto_Tensor& t) {
    if (!t.has_shape()) return "[]";
    std::string s = "[";
    for (int i = 0; i < t.shape().dim_size(); ++i) {
        if (i) s += ", ";
        const auto& d = t.shape().dim(i);
        if (d.has_dim_value()) s += std::to_string(d.dim_value());
        else if (d.has_dim_param()) s += d.dim_param();
        else s += "?";
    }
    return s + "]";
}

static void print_value_info(const onnx::ValueInfoProto& v) {
    std::cout << "  " << v.name();
    if (v.type().has_tensor_type()) {
        std::cout << "  shape=" << shape_str(v.type().tensor_type());
    }
    std::cout << "\n";
}

static void print_attr(const onnx::AttributeProto& attr) {
    std::cout << "      " << attr.name() << "=";
    switch (attr.type()) {
        case onnx::AttributeProto::INT:
            std::cout << attr.i();
            break;
        case onnx::AttributeProto::FLOAT:
            std::cout << attr.f();
            break;
        case onnx::AttributeProto::STRING:
            std::cout << "\"" << attr.s() << "\"";
            break;
        case onnx::AttributeProto::INTS: {
            std::cout << "[";
            for (int i = 0; i < attr.ints_size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << attr.ints(i);
            }
            std::cout << "]";
            break;
        }
        case onnx::AttributeProto::FLOATS: {
            std::cout << "[";
            for (int i = 0; i < attr.floats_size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << attr.floats(i);
            }
            std::cout << "]";
            break;
        }
        default:
            std::cout << "<type=" << attr.type() << ">";
            break;
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: print_onnx <model.onnx>\n";
        return 1;
    }

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open: " << argv[1] << "\n";
        return 1;
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    onnx::ModelProto model;
    if (!model.ParseFromString(content)) {
        std::cerr << "Failed to parse ONNX model\n";
        return 1;
    }

    // Model metadata
    std::cout << "=== Model Metadata ===\n";
    std::cout << "  IR version: " << model.ir_version() << "\n";
    if (model.opset_import_size() > 0) {
        const auto& op = model.opset_import(0);
        std::cout << "  Opset: version=" << op.version()
                  << " domain=\"" << op.domain() << "\"\n";
    }
    std::cout << "\n";

    const auto& g = model.graph();

    // Graph inputs
    std::cout << "=== Graph Inputs (" << g.input_size() << ") ===\n";
    for (const auto& inp : g.input()) print_value_info(inp);
    std::cout << "\n";

    // Graph outputs
    std::cout << "=== Graph Outputs (" << g.output_size() << ") ===\n";
    for (const auto& out : g.output()) print_value_info(out);
    std::cout << "\n";

    // Initializers (weights)
    std::cout << "=== Initializers (" << g.initializer_size() << ") ===\n";
    for (const auto& init : g.initializer()) {
        std::cout << "  " << init.name()
                  << "  dtype=" << init.data_type() << "  dims=[";
        for (int i = 0; i < init.dims_size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << init.dims(i);
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    // Nodes
    std::cout << "=== Nodes (" << g.node_size() << ") ===\n";
    for (int i = 0; i < g.node_size(); ++i) {
        const auto& n = g.node(i);
        std::cout << "[" << i << "] op=" << n.op_type();
        if (!n.name().empty()) std::cout << " name=" << n.name();
        std::cout << "\n";

        std::cout << "    inputs:  [";
        for (int j = 0; j < n.input_size(); ++j) {
            if (j) std::cout << ", ";
            std::cout << n.input(j);
        }
        std::cout << "]\n";

        std::cout << "    outputs: [";
        for (int j = 0; j < n.output_size(); ++j) {
            if (j) std::cout << ", ";
            std::cout << n.output(j);
        }
        std::cout << "]\n";

        if (n.attribute_size() > 0) {
            std::cout << "    attributes:\n";
            for (const auto& attr : n.attribute()) print_attr(attr);
        }
    }

    return 0;
}
