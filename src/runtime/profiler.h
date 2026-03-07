#pragma once
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace tinyinfer {

struct OpTiming {
    std::string op_name;   // e.g., "Gemm/fc1"
    float elapsed_ms = 0.f;
};

class Profiler {
public:
    Profiler();
    ~Profiler();

    // Call before launching the kernel for 'op_name'
    void start(const std::string& op_name, cudaStream_t stream);

    // Call immediately after the kernel launch
    void stop(cudaStream_t stream);

    // Block until stream completes, compute elapsed times, return results
    std::vector<OpTiming> report(cudaStream_t stream);

    void reset();

private:
    struct EventPair {
        std::string op_name;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop  = nullptr;
    };
    std::vector<EventPair> events_;
};

}  // namespace tinyinfer
