#include "runtime/profiler.h"
#include <stdexcept>

namespace tinyinfer {

Profiler::Profiler() = default;

Profiler::~Profiler() {
    reset();
}

void Profiler::start(const std::string& op_name, cudaStream_t stream) {
    EventPair ep;
    ep.op_name = op_name;
    cudaEventCreate(&ep.start);
    cudaEventCreate(&ep.stop);
    cudaEventRecord(ep.start, stream);
    events_.push_back(std::move(ep));
}

void Profiler::stop(cudaStream_t stream) {
    if (events_.empty()) return;
    cudaEventRecord(events_.back().stop, stream);
}

std::vector<OpTiming> Profiler::report(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    std::vector<OpTiming> results;
    results.reserve(events_.size());
    for (auto& ep : events_) {
        OpTiming t;
        t.op_name = ep.op_name;
        cudaEventElapsedTime(&t.elapsed_ms, ep.start, ep.stop);
        results.push_back(t);
    }
    return results;
}

void Profiler::reset() {
    for (auto& ep : events_) {
        if (ep.start) cudaEventDestroy(ep.start);
        if (ep.stop)  cudaEventDestroy(ep.stop);
    }
    events_.clear();
}

}  // namespace tinyinfer
