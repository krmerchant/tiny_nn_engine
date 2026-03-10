#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

namespace tinyinfer {

enum class DataType { Float32 };


class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const TensorDesc& desc);

    // Factory: allocate on device and copy from host data
    static Tensor from_host(const std::vector<float>& data, const std::vector<int64_t>& shape);

    // Copy device data back to host
    std::vector<float> to_host() const;

    // argmax along a given axis (currently supports axis=1 for 2-D tensors)
    std::vector<int> argmax(int axis = 1) const;

    // Raw pointer to device memory (read-only)
    const float* data() const { return data_.get(); }
    float* data_mut() { return data_.get(); }

    const TensorDesc& desc() const { return desc_; }
    const std::vector<int64_t>& shape() const { return desc_.shape; }
    int64_t numel() const { return desc_.numel(); }
    bool empty() const { return !data_; }

private:
    TensorDesc desc_;
    std::vector<int64_t> shape;
    DataType dtype = DataType::Float32;


    std::shared_ptr<float> data_;  // device memory, freed with cudaFree

    static std::shared_ptr<float> alloc_device(size_t n_floats);
};

}  // namespace tinyinfer
