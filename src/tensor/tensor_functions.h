#pragma once
#include "tensor/tensor.h"
#include <cstring>
#include <cuda_runtime.h>

namespace tinyinfer {

// Returns a tensor of the given shape with all elements set to 0
inline Tensor zeros(const std::vector<int64_t>& shape) {
    Tensor t(shape);
    int64_t n = 1;
    for (int64_t d : shape) n *= d;

    if (t.data_ptr()) {
        // Tensor defaults to CPU; if moved to GPU before calling zeros, use cudaMemset
        // For now assumes CPU allocation
        std::memset(t.data_ptr(), 0, n * sizeof(float));
    }
    return t;
}

}  // namespace tinyinfer
