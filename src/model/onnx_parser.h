#pragma once
#include "tinyinfer/graph.h"
#include <string>

namespace tinyinfer {
namespace internal {

// Parse an ONNX model file, upload initializer tensors to GPU, and return a
// topologically-sorted Graph. Throws std::runtime_error on failure.
Graph parse_onnx(const std::string& path);

}  // namespace internal
}  // namespace tinyinfer
