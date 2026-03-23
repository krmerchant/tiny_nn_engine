#include "tensor/tensor.h"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
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

    // Add bias to every row of out (each row is N elements)
    const float one = 1.0f;
    for (int m = 0; m < M; ++m)
        cublasSaxpy(cublas_handle_, N, &one, bias, 1, out + m * N, 1);
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

// One block per row; blockDim.x must be a power of 2 (we use 32).
// Shared memory: blockDim.x floats for reduction scratch.
__global__ static void softmax_kernel(const float* in, float* out, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    const float* row_in  = in  + row * cols;
    float*       row_out = out + row * cols;

    // Step 1: thread-local max
    float local_max = -1e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row_in[i];
        if (v > local_max) local_max = v;
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();

    // Reduce to global row max
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] = smem[threadIdx.x] > smem[threadIdx.x + stride]
                                 ? smem[threadIdx.x] : smem[threadIdx.x + stride];
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Step 2: exp and partial sum
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = __expf(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce to global sum
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();

    // Step 3: normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        row_out[i] /= row_sum;
}

void GPUStorage::softmax(const float* in, float* out, int rows, int cols,
                         cudaStream_t stream) const {
    int threads = 32;  // one warp; power-of-2, sufficient for typical col counts
    size_t smem  = threads * sizeof(float);
    softmax_kernel<<<rows, threads, smem, stream>>>(in, out, cols);
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

void CPUStorage::softmax(const float* in, float* out, int rows, int cols, cudaStream_t) const {
    for (int r = 0; r < rows; ++r) {
        const float* row_in  = in  + r * cols;
        float*       row_out = out + r * cols;

        float max_val = row_in[0];
        for (int c = 1; c < cols; ++c)
            if (row_in[c] > max_val) max_val = row_in[c];

        float sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            row_out[c] = std::exp(row_in[c] - max_val);
            sum += row_out[c];
        }
        for (int c = 0; c < cols; ++c)
            row_out[c] /= sum;
    }
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
// clone() — deep copy to the same device
// ---------------------------------------------------------------------------

Tensor Tensor::clone() const {
    int64_t n = num_elements(_shape);
    Tensor result;
    result._shape   = _shape;
    result._dtype   = _dtype;
    result._storage = _storage->make_empty();
    result._data    = result._storage->alloc(n);
    if (device() == Device::GPU)
        cudaMemcpy(result._data, _data, n * sizeof(float), cudaMemcpyDeviceToDevice);
    else
        std::memcpy(result._data, _data, n * sizeof(float));
    return result;
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
// softmax() — row-wise softmax on a 2-D tensor
// ---------------------------------------------------------------------------

Tensor Tensor::softmax(cudaStream_t stream) const {
    if (_shape.size() != 2)
        throw std::runtime_error("Tensor::softmax: only 2-D tensors supported");
    int rows = (int)_shape[0];
    int cols = (int)_shape[1];
    Tensor out(_shape);
    if (device() == Device::GPU) out.cuda();
    _storage->softmax(data_ptr(), out.data_ptr(), rows, cols, stream);
    return out;
}

// ---------------------------------------------------------------------------
// reshape_() — in-place shape change (data layout unchanged)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// to_string() — human-readable tensor contents
// ---------------------------------------------------------------------------

std::string Tensor::to_string() const {
    if (empty()) return "tensor(<empty>)";

    std::ostringstream os;
    os << std::fixed << std::setprecision(4);

    auto read = [&](int64_t i) { return _storage->read_element(_data, i); };
    int64_t n = num_elements(_shape);

    if (_shape.size() == 1) {
        int64_t show = n <= 16 ? n : 8;
        os << "[";
        for (int64_t i = 0; i < show; ++i) {
            if (i) os << ", ";
            os << read(i);
        }
        if (n > 16) os << ", ...";
        os << "]";
    } else if (_shape.size() == 2) {
        int64_t rows = _shape[0], cols = _shape[1];
        int64_t show_rows = rows <= 3 ? rows : 2;
        int64_t show_cols = cols <= 8 ? cols : 4;
        os << "[";
        for (int64_t r = 0; r < show_rows; ++r) {
            if (r) os << ", ";
            os << "[";
            for (int64_t c = 0; c < show_cols; ++c) {
                if (c) os << ", ";
                os << read(r * cols + c);
            }
            if (cols > 8) os << ", ...";
            os << "]";
        }
        if (rows > 3) os << ", ...";
        os << "]";
    } else {
        int64_t show = n <= 16 ? n : 8;
        os << "[";
        for (int64_t i = 0; i < show; ++i) {
            if (i) os << ", ";
            os << read(i);
        }
        if (n > 16) os << ", ...";
        os << "]";
    }

    return "tensor(" + shape_str() + "): " + os.str();
}

// ---------------------------------------------------------------------------
// to_matlab() — write tensor to a .m file for MATLAB plotting
// ---------------------------------------------------------------------------

void Tensor::to_matlab(const std::string& path, const std::string& var_name) const {
    if (_shape.size() > 2)
        throw std::runtime_error("Tensor::to_matlab: only 1-D and 2-D tensors supported");

    std::ofstream f(path);
    if (!f) throw std::runtime_error("Tensor::to_matlab: cannot open file: " + path);

    f << std::fixed << std::setprecision(4);
    f << "% tensor(" << shape_str() << ")\n";
    f << var_name << " = [";

    int64_t rows = (_shape.size() == 2) ? _shape[0] : 1;
    int64_t cols = (_shape.size() == 2) ? _shape[1] : _shape[0];

    for (int64_t r = 0; r < rows; ++r) {
        if (r) f << "; ";
        for (int64_t c = 0; c < cols; ++c) {
            if (c) f << ", ";
            f << _storage->read_element(_data, r * cols + c);
        }
    }
    f << "];\n";
}

// ---------------------------------------------------------------------------
// reshape_() — in-place shape change (data layout unchanged)
// ---------------------------------------------------------------------------

void Tensor::reshape_(const std::vector<int64_t>& new_shape) {
    int64_t old_n = num_elements(_shape);
    int64_t new_n = 1;
    for (int64_t d : new_shape) new_n *= d;
    if (old_n != new_n)
        throw std::runtime_error("Tensor::reshape_: element count mismatch");
    _shape = new_shape;
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
