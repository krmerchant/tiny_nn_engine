#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

namespace tinyinfer {

enum class DataType { Float32 };
enum class Device { GPU, CPU };


class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const std::vector<int64_t>);
    ~Tensor();

    // Factory: allocate on device and copy from host data
    Tensor& cuda();

    // Move underlying data to CPU if not already there
    Tensor& cpu();

// Raw pointer to device memory (read-only)
    const float* data_ptr() const { return _data; }

    const std::vector<int64_t>& shape() const { return _shape; }
    bool empty() const { return !_data; }

private:
    std::vector<int64_t> _shape;
    DataType _dtype = DataType::Float32;
    Device _device = Device::CPU;

    float* _data = nullptr;  // device or host memory depending on _device

};

}  // namespace tinyinfer
