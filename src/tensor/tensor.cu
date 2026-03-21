#include "tensor/tensor.h"
#include <stdexcept>

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

namespace internal {

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
    cudaMemcpy(out, a, n * sizeof(float), cudaMemcpyDeviceToDevice);
    const float alpha = 1.0f;
    cublasSaxpy(cublas_handle_, static_cast<int>(n), &alpha, b, 1, out, 1);
}

// out(M,N) = self(M,K) * weight(N,K)^T + bias(N,)
//
// Row-major trick: row-major X(r,c) looks like col-major X^T(c,r) to cuBLAS.
// We want out^T(N,M) = weight(N,K) * self(M,K)^T
//   cuBLAS sees weight ptr as weight^T(K,N) → CUBLAS_OP_T gives weight(N,K)
//   cuBLAS sees self   ptr as self^T(K,M)   → CUBLAS_OP_N keeps self^T as-is
//
// TODO: revisit handle-per-GPUStorage design. Each tensor currently owns its
// own cublasHandle_t. A real engine would share one handle across the entire
// session (owned by the Executor) to avoid per-tensor overhead and allow
// stream binding at the executor level.
void GPUStorage::gemm(const float* A, const float* B, const float* bias,
                      float* out, int M, int N, int K) const {
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B,   K,   // weight^T: K×N col-major
        A,   K,   // self^T:   K×M col-major
        &beta,
        out, N);  // out^T:    N×M col-major

    // Add bias row: out[0..N-1] += bias[0..N-1]  (correct for M=1)
    const float one = 1.0f;
    cublasSaxpy(cublas_handle_, N, &one, bias, 1, out, 1);
}

__global__ static void relu_kernel(const float* in, float* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > 0.f ? in[i] : 0.f;
}

void GPUStorage::relu(const float* in, float* out, int64_t n, cudaStream_t stream) const {
    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);
    relu_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

std::unique_ptr<TensorStorage> GPUStorage::make_empty() const {
    return std::make_unique<GPUStorage>();
}

// ---------------------------------------------------------------------------
// CPUStorage implementation
// ---------------------------------------------------------------------------

std::unique_ptr<TensorStorage> CPUStorage::make_empty() const {
    return std::make_unique<CPUStorage>();
}

void CPUStorage::gemm(const float* A, const float* B, const float* bias,
                      float* out, int M, int N, int K) const {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = bias[n];
            for (int k = 0; k < K; ++k)
                acc += A[m*K + k] * B[n*K + k];  // B stored as N×K
            out[m*N + n] = acc;
        }
}

void CPUStorage::relu(const float* in, float* out, int64_t n, cudaStream_t) const {
    for (int64_t i = 0; i < n; ++i) out[i] = in[i] > 0.f ? in[i] : 0.f;
}

}  // namespace internal

// ---------------------------------------------------------------------------
// Tensor constructors
// ---------------------------------------------------------------------------

Tensor::Tensor(const std::vector<int64_t> shape) : _shape(shape), _data(nullptr) {
    _storage = std::make_unique<internal::CPUStorage>();
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
    _storage = std::make_unique<internal::CPUStorage>();
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
    auto gpu = std::make_unique<internal::GPUStorage>();
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
    auto cpu = std::make_unique<internal::CPUStorage>();
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
// gemm() — out(M,N) = self(M,K) * weight(N,K)^T + bias(N,)
// ---------------------------------------------------------------------------

Tensor Tensor::gemm(const Tensor& weight, const Tensor& bias) const {
    int M = (int)_shape[0];
    int K = (int)_shape[1];
    int N = (int)weight.shape()[0];

    Tensor out({(int64_t)M, (int64_t)N});
    if (device() == Device::GPU) out.cuda();

    _storage->gemm(data_ptr(), weight.data_ptr(), bias.data_ptr(),
                   out.data_ptr(), M, N, K);
    return out;
}

// ---------------------------------------------------------------------------
// relu() — element-wise max(0, x)
// ---------------------------------------------------------------------------

Tensor Tensor::relu(cudaStream_t stream) const {
    int64_t n = num_elements(_shape);
    Tensor out(_shape);
    if (device() == Device::GPU) out.cuda();
    _storage->relu(data_ptr(), out.data_ptr(), n, stream);
    return out;
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
