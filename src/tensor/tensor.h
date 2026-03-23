#pragma once
#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace tinyinfer {

enum class DataType { Float32 };
enum class Device { CPU, GPU };

// ---------------------------------------------------------------------------
// TensorStorage — abstract device backend (internal)
// ---------------------------------------------------------------------------

namespace internal {

class TensorStorage {
public:
    virtual float* alloc(int64_t n) = 0;
    virtual void   dealloc(float* p) = 0;
    virtual void   fill(float* p, int64_t n, float val) = 0;
    virtual float  read_element(const float* p, int64_t offset) const = 0;
    virtual void   add(const float* a, const float* b, float* out, int64_t n) const = 0;
    virtual void   gemm(const float* A, const float* B, const float* bias,
                        float* out, int M, int N, int K) const = 0;
    virtual void   relu(const float* in, float* out, int64_t n, cudaStream_t stream) const = 0;
    virtual void   softmax(const float* in, float* out, int rows, int cols, cudaStream_t stream) const = 0;
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
    void   gemm(const float* A, const float* B, const float* bias,
                float* out, int M, int N, int K) const override;
    void   relu(const float* in, float* out, int64_t n, cudaStream_t stream) const override;
    void   softmax(const float* in, float* out, int rows, int cols, cudaStream_t stream) const override;
    std::unique_ptr<TensorStorage> make_empty() const override;
    Device device() const override { return Device::CPU; }
};

class GPUStorage : public TensorStorage {
public:
    GPUStorage()  { cublasCreate(&cublas_handle_); }
    ~GPUStorage() { cublasDestroy(cublas_handle_); }

    float* alloc(int64_t n) override;
    void   dealloc(float* p) override;
    void   fill(float* p, int64_t n, float val) override;
    float  read_element(const float* p, int64_t i) const override;
    void   add(const float* a, const float* b, float* out, int64_t n) const override;
    void   gemm(const float* A, const float* B, const float* bias,
                float* out, int M, int N, int K) const override;
    void   relu(const float* in, float* out, int64_t n, cudaStream_t stream) const override;
    void   softmax(const float* in, float* out, int rows, int cols, cudaStream_t stream) const override;
    std::unique_ptr<TensorStorage> make_empty() const override;
    Device device() const override { return Device::GPU; }

private:
    cublasHandle_t cublas_handle_;
};

}  // namespace internal

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

    // Deep copy to the same device (D2D cudaMemcpy for GPU, memcpy for CPU)
    Tensor clone() const;

    Tensor operator+(const Tensor& other) const;

    // Returns new GPU tensor: out(M,N) = self(M,K) * weight(N,K)^T + bias(N,)
    // self, weight, and bias must already be on GPU
    Tensor gemm(const Tensor& weight, const Tensor& bias) const;

    // Returns new tensor with ReLU applied element-wise; output on same device as self
    Tensor relu(cudaStream_t stream = nullptr) const;

    // Returns new tensor with row-wise softmax applied; self must be 2-D
    Tensor softmax(cudaStream_t stream = nullptr) const;

    // In-place reshape: updates shape metadata without touching data
    void reshape_(const std::vector<int64_t>& new_shape);

    const std::vector<int64_t>& shape() const { return _shape; }
    std::string shape_str() const {
        std::string s = "[";
        for (size_t i = 0; i < _shape.size(); ++i) {
            if (i) s += ", ";
            s += std::to_string(_shape[i]);
        }
        return s + "]";
    }
    std::string to_string() const;
    void to_matlab(const std::string& path, const std::string& var_name = "data") const;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        return os << t.to_string();
    }
    bool empty() const { return !_data; }
    Device device() const { return _storage ? _storage->device() : Device::CPU; }

private:
    std::vector<int64_t> _shape;
    DataType _dtype = DataType::Float32;
    std::unique_ptr<internal::TensorStorage> _storage;
    float* _data = nullptr;


};

}  // namespace tinyinfer
