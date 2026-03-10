#include "tensor/tensor.h"

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
    // TODO: cudaMalloc(&dev, n * sizeof(float));
    // TODO: cudaMemcpy(dev, _data, n * sizeof(float), cudaMemcpyHostToDevice);
    // TODO: delete[] _data;
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


}  // namespace tinyinfer
