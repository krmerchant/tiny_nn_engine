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

    // Element access by multi-dimensional index (row-major)
    float at(const std::vector<uint>& idx) const;

    const float* data_ptr() const { return _data; }
    float* data_ptr() { return _data; }

    void fill(float val);

    Tensor operator+(const Tensor& other) const;

    const std::vector<int64_t>& shape() const { return _shape; }
    bool empty() const { return !_data; }

private:
    std::vector<int64_t> _shape;
    DataType _dtype = DataType::Float32;
    Device _device = Device::CPU;

    float* _data = nullptr;  // device or host memory depending on _device

};

}  // namespace tinyinfer
