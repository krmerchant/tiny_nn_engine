#include "runtime/executor.h"
#include "ops/op_registry.h"
#include <chrono>
#include <iomanip>
#include <iostream>

namespace tinyinfer {

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
// GPUExecutor
// ---------------------------------------------------------------------------

GPUExecutor::~GPUExecutor() {
    if (stream_) cudaStreamDestroy(stream_);
}

std::unordered_map<std::string, Tensor>
GPUExecutor::run(std::unordered_map<std::string, Tensor> inputs) {
    const Graph& g = model_->graph();

    // 0. Clear stale intermediates
    for (const Node& node : g.nodes)
        for (const auto& out_name : node.outputs)
            value_map_.erase(out_name);

    // Helper: record a wall-clock + GPU-event timed phase
    auto record = [&](const std::string& name, auto fn) {
        if (!enable_profiling_) { fn(); return; }
        if (phase_stats_.find(name) == phase_stats_.end()) phase_order_.push_back(name);
        profiler_.start(name, stream_);
        auto t0 = Clock::now();
        fn();
        double wall = Ms(Clock::now() - t0).count();
        profiler_.stop(stream_);
        // GPU time resolved later in report(); store wall now
        phase_stats_[name].wall_ms += wall;
        phase_stats_[name].count++;
    };

    // 1. Upload inputs to GPU
    record("upload", [&]() {
        for (auto& [name, tensor] : inputs) {
            tensor.cuda();
            value_map_[name] = std::move(tensor);
        }
    });

    // 2. Dispatch ops
    for (const Node& node : g.nodes) {
        std::string label = node.name.empty() ? node.outputs[0] : node.name;
        record(label, [&]() {
            auto kernel = internal::OpRegistry::instance().create(node.op_type);
            internal::KernelContext ctx{node, value_map_, stream_};
            kernel->execute(ctx, value_map_);
        });
    }

    // 3. Sync stream
    record("stream_sync", [&]() { cudaStreamSynchronize(stream_); });

    // Accumulate GPU event times after sync
    if (enable_profiling_) {
        auto timings = profiler_.report(stream_);
        for (const auto& t : timings)
            phase_stats_[t.op_name].gpu_ms += t.elapsed_ms;
        profiler_.reset();
    }

    // 4. Collect and erase outputs
    std::unordered_map<std::string, Tensor> outputs;
    for (const auto& name : g.output_names) {
        outputs[name] = std::move(value_map_.at(name));
        value_map_.erase(name);
    }
    return outputs;
}

void GPUExecutor::print_profile(int num_samples) const {
    if (phase_stats_.empty()) {
        std::cout << "[profiler] No data — enable_profiling was not set.\n";
        return;
    }
    const int n = num_samples > 0 ? num_samples : 1;
    std::cout << "\n=== GPUExecutor Profile (avg over " << n << " samples) ===\n";
    std::cout << std::left << std::setw(28) << "Phase"
              << std::right << std::setw(12) << "wall ms"
              << std::setw(12) << "GPU ms"
              << std::setw(14) << "overhead ms" << "\n";
    std::cout << std::string(66, '-') << "\n";

    double total_wall = 0, total_gpu = 0;
    for (const auto& name : phase_order_) {
        const auto& s = phase_stats_.at(name);
        double wall = s.wall_ms / n;
        double gpu  = s.gpu_ms  / n;
        double ovhd = wall - gpu;
        total_wall += wall;
        total_gpu  += gpu;
        std::cout << std::left  << std::setw(28) << name
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(12) << wall
                  << std::setw(12) << gpu
                  << std::setw(14) << ovhd << "\n";
    }
    std::cout << std::string(66, '-') << "\n";
    std::cout << std::left  << std::setw(28) << "TOTAL"
              << std::right << std::setw(12) << total_wall
              << std::setw(12) << total_gpu
              << std::setw(14) << (total_wall - total_gpu) << "\n";
}

// ---------------------------------------------------------------------------
// CPUExecutor
// ---------------------------------------------------------------------------

std::unordered_map<std::string, Tensor>
CPUExecutor::run(std::unordered_map<std::string, Tensor> inputs) {
    const Graph& g = model_->graph();

    // 0. Clear stale intermediates
    for (const Node& node : g.nodes)
        for (const auto& out_name : node.outputs)
            value_map_.erase(out_name);

    // 1. Insert inputs (CPU, no transfer)
    for (auto& [name, tensor] : inputs)
        value_map_[name] = std::move(tensor);

    // 2. Dispatch ops
    for (const Node& node : g.nodes) {
        auto kernel = internal::OpRegistry::instance().create(node.op_type);
        internal::KernelContext ctx{node, value_map_, /*stream=*/nullptr};
        kernel->execute(ctx, value_map_);
    }

    // 3. Collect and erase outputs
    std::unordered_map<std::string, Tensor> outputs;
    for (const auto& name : g.output_names) {
        outputs[name] = std::move(value_map_.at(name));
        value_map_.erase(name);
    }
    return outputs;
}

}  // namespace tinyinfer
