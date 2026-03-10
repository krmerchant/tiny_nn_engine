#include "tensor/tensor.h"
#include <stdexcept>
#include <algorithm>
#include <cublas_v2.h>

namespace tinyinfer {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int64_t num_elements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Tensor::Tensor(const std::vector<int64_t> shape) : _shape(shape), _data(nullptr) {
    int64_t n = num_elements(_shape);
    if (_device == Device::CPU) {
        _data = new float[n];
    } else {
        cudaMalloc(&_data, n * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

Tensor::~Tensor() {
    if (!_data) return;
    if (_device == Device::CPU) {
        delete[] _data;
    } else {
        cudaFree(_data);
    }
    _data = nullptr;
}

// ---------------------------------------------------------------------------
// cuda() — move underlying data to GPU if not already there
// ---------------------------------------------------------------------------

Tensor& Tensor::cuda() {
    if (_device == Device::GPU) return *this;

    int64_t n = num_elements(_shape);
    float* dev = nullptr;
    cudaMalloc(&dev, n * sizeof(float));
    cudaMemcpy(dev, _data, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] _data;
    _data = dev;
    _device = Device::GPU;
    return *this;
}

// ---------------------------------------------------------------------------
// cpu() — move underlying data to CPU if not already there
// ---------------------------------------------------------------------------

Tensor& Tensor::cpu() {
    if (_device == Device::CPU) return *this;

    int64_t n = num_elements(_shape);
    float* host = new float[n];
    // TODO: cudaMemcpy(host, _data, n * sizeof(float), cudaMemcpyDeviceToHost);
    // TODO: cudaFree(_data);
    _data = host;
    _device = Device::CPU;
    return *this;
}


// ---------------------------------------------------------------------------
// fill() — set all elements to a given value
// ---------------------------------------------------------------------------

void Tensor::fill(float val) {
    int64_t n = num_elements(_shape);
    if (_device == Device::CPU) {
        std::fill(_data, _data + n, val);
    } else {
        // TODO: launch a CUDA kernel to fill device buffer
    }
}

// ---------------------------------------------------------------------------
// operator+ — element-wise addition
// ---------------------------------------------------------------------------

Tensor Tensor::operator+(const Tensor& other) const {
    if (_shape != other._shape)
        throw std::runtime_error("Tensor::operator+: shape mismatch");
    if (other._device != this->_device)
        throw std::runtime_error("Tensor::operator+: tensors on different devices");

    int64_t n = num_elements(_shape);
    Tensor result(_shape);

    if (_device == Device::CPU) {
        for (int64_t i = 0; i < n; ++i)
            result._data[i] = _data[i] + other._data[i];
    } else {
        result.cuda();  // allocate result buffer on GPU before cuBLAS operations
        // Copy this tensor's data into result on GPU
        cudaMemcpy(result._data, _data, n * sizeof(float), cudaMemcpyDeviceToDevice);

        // result = 1.0 * other + result  (i.e. result = this + other)
        cublasHandle_t handle;
        cublasCreate(&handle);
        const float alpha = 1.0f;
        cublasSaxpy(handle, static_cast<int>(n), &alpha, other._data, 1, result._data, 1);
        cublasDestroy(handle);
    }

    return result;
}

// ---------------------------------------------------------------------------
// at() — element access by multi-dimensional index (row-major)
// ---------------------------------------------------------------------------

float Tensor::at(const std::vector<uint>& idx) const {
    if (idx.size() != _shape.size())
        throw std::out_of_range("Tensor::at: number of indices does not match number of dimensions");

    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= static_cast<uint>(_shape[i]))
            throw std::out_of_range("Tensor::at: index out of range");
    }

    // compute flat row-major offset
    int64_t flat = 0;
    int64_t stride = 1;
    for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i) {
        flat += idx[i] * stride;
        stride *= _shape[i];
    }

    if (_device == Device::CPU) {
        return _data[flat];
    } else {
        float val;
        cudaMemcpy(&val, _data + flat, sizeof(float), cudaMemcpyDeviceToHost);
        return val;
    }
}

}  // namespace tinyinfer
