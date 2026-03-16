#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

namespace tinyinfer {

enum class DataType { Float32 };
enum class Device { CPU, GPU };

// ---------------------------------------------------------------------------
// TensorStorage — abstract device backend
// ---------------------------------------------------------------------------

class TensorStorage {
public:
    virtual float* alloc(int64_t n) = 0;
    virtual void   dealloc(float* p) = 0;
    virtual void   fill(float* p, int64_t n, float val) = 0;
    virtual float  read_element(const float* p, int64_t offset) const = 0;
    virtual void   add(const float* a, const float* b, float* out, int64_t n) const = 0;
    virtual std::unique_ptr<TensorStorage> make_empty() const = 0;
    virtual Device device() const = 0;
    virtual ~TensorStorage() = default;
};

class CPUStorage : public TensorStorage {
public:
    float* alloc(int64_t n) override { return new float[n]; }
    void   dealloc(float* p) override { delete[] p; }
    void   fill(float* p, int64_t n, float val) override { std::fill(p, p + n, val); }
    float  read_element(const float* p, int64_t i) const override { return p[i]; }
    void   add(const float* a, const float* b, float* out, int64_t n) const override {
        for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }
    std::unique_ptr<TensorStorage> make_empty() const override;
    Device device() const override { return Device::CPU; }
};

class GPUStorage : public TensorStorage {
public:
    float* alloc(int64_t n) override;
    void   dealloc(float* p) override;
    void   fill(float* p, int64_t n, float val) override;
    float  read_element(const float* p, int64_t i) const override;
    void   add(const float* a, const float* b, float* out, int64_t n) const override;
    std::unique_ptr<TensorStorage> make_empty() const override;
    Device device() const override { return Device::GPU; }
};

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const std::vector<int64_t> shape);
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Move underlying data to GPU if not already there
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
    Device device() const { return _storage ? _storage->device() : Device::CPU; }

private:
    std::vector<int64_t> _shape;
    DataType _dtype = DataType::Float32;
    std::unique_ptr<TensorStorage> _storage;
    float* _data = nullptr;


};

}  // namespace tinyinfer
