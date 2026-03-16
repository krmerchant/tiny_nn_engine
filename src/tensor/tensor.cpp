#include "tensor/tensor.h"
#include <stdexcept>
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
// GPUStorage implementation
// ---------------------------------------------------------------------------

float* GPUStorage::alloc(int64_t n) {
    float* p;
    cudaMalloc(&p, n * sizeof(float));
    return p;
}

void GPUStorage::dealloc(float* p) {
    cudaFree(p);
}

float GPUStorage::read_element(const float* p, int64_t i) const {
    float v;
    cudaMemcpy(&v, p + i, sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

void GPUStorage::fill(float* /*p*/, int64_t /*n*/, float /*val*/) {
    // TODO: launch a CUDA kernel to fill device buffer
}

void GPUStorage::add(const float* a, const float* b, float* out, int64_t n) const {
    // out = a + b via cuBLAS: copy a into out, then out += 1*b
    cudaMemcpy(out, a, n * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    cublasSaxpy(handle, static_cast<int>(n), &alpha, b, 1, out, 1);
    cublasDestroy(handle);
}

std::unique_ptr<TensorStorage> GPUStorage::make_empty() const {
    return std::make_unique<GPUStorage>();
}

// ---------------------------------------------------------------------------
// CPUStorage make_empty (defined in .cpp to avoid incomplete-type issues)
// ---------------------------------------------------------------------------

std::unique_ptr<TensorStorage> CPUStorage::make_empty() const {
    return std::make_unique<CPUStorage>();
}

// ---------------------------------------------------------------------------
// Tensor constructors
// ---------------------------------------------------------------------------

Tensor::Tensor(const std::vector<int64_t> shape) : _shape(shape), _data(nullptr) {
    _storage = std::make_unique<CPUStorage>();
    _data = _storage->alloc(num_elements(_shape));
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape)
    : _shape(shape), _data(nullptr)
{
    int64_t n = num_elements(_shape);
    if (static_cast<int64_t>(data.size()) != n) {
        throw std::runtime_error(
            "Tensor: data size " + std::to_string(data.size()) +
            " does not match shape product " + std::to_string(n));
    }
    _storage = std::make_unique<CPUStorage>();
    _data = _storage->alloc(n);
    std::copy(data.begin(), data.end(), _data);
}

// ---------------------------------------------------------------------------
// Move constructor / move assignment
// ---------------------------------------------------------------------------

Tensor::Tensor(Tensor&& other) noexcept
    : _shape(std::move(other._shape)), _dtype(other._dtype),
      _storage(std::move(other._storage)), _data(other._data)
{
    other._data = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (_data && _storage) _storage->dealloc(_data);
        _shape   = std::move(other._shape);
        _dtype   = other._dtype;
        _storage = std::move(other._storage);
        _data    = other._data;
        other._data = nullptr;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

Tensor::~Tensor() {
    if (!_data || !_storage) return;
    _storage->dealloc(_data);
    _data = nullptr;
}

// ---------------------------------------------------------------------------
// cuda() — move underlying data to GPU if not already there
// ---------------------------------------------------------------------------

Tensor& Tensor::cuda() {
    if (device() == Device::GPU) return *this;

    int64_t n = num_elements(_shape);
    auto gpu = std::make_unique<GPUStorage>();
    float* dev = gpu->alloc(n);
    cudaMemcpy(dev, _data, n * sizeof(float), cudaMemcpyHostToDevice);
    _storage->dealloc(_data);
    _storage = std::move(gpu);
    _data = dev;
    return *this;
}

// ---------------------------------------------------------------------------
// cpu() — move underlying data to CPU if not already there
// ---------------------------------------------------------------------------

Tensor& Tensor::cpu() {
    if (device() == Device::CPU) return *this;

    int64_t n = num_elements(_shape);
    auto cpu = std::make_unique<CPUStorage>();
    float* host = cpu->alloc(n);
    cudaMemcpy(host, _data, n * sizeof(float), cudaMemcpyDeviceToHost);
    _storage->dealloc(_data);
    _storage = std::move(cpu);
    _data = host;
    return *this;
}

// ---------------------------------------------------------------------------
// fill() — set all elements to a given value
// ---------------------------------------------------------------------------

void Tensor::fill(float val) {
    _storage->fill(_data, num_elements(_shape), val);
}

// ---------------------------------------------------------------------------
// operator+ — element-wise addition
// ---------------------------------------------------------------------------

Tensor Tensor::operator+(const Tensor& other) const {
    if (_shape != other._shape)
        throw std::runtime_error("operator+: shape mismatch");
    if (other.device() != device())
        throw std::runtime_error("operator+: device mismatch");

    int64_t n = num_elements(_shape);

    Tensor result;
    result._storage = _storage->make_empty();
    result._shape   = _shape;
    result._data    = result._storage->alloc(n);

    _storage->add(_data, other._data, result._data, n);
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

    int64_t flat = 0;
    int64_t stride = 1;
    for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i) {
        flat += idx[i] * stride;
        stride *= _shape[i];
    }

    return _storage->read_element(_data, flat);
}

}  // namespace tinyinfer
