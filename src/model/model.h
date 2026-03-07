#pragma once
#include <string>
#include "tinyinfer/graph.h"

namespace tinyinfer {

class Model {
public:
    // Parse ONNX file, upload initializers to GPU, build topologically-sorted Graph
    static Model load(const std::string& onnx_path);

    // Print node names, op types, and input/output tensor names
    void print_graph() const;

    const Graph& graph() const { return graph_; }

private:
    explicit Model(Graph g) : graph_(std::move(g)) {}
    Graph graph_;
};

}  // namespace tinyinfer
