#include "runtime/executor.h"
#include "ops/op_registry.h"

namespace tinyinfer {

GPUExecutor::~GPUExecutor() {
    if (stream_) cudaStreamDestroy(stream_);
}

std::unordered_map<std::string, Tensor>
GPUExecutor::run(std::unordered_map<std::string, Tensor> inputs) {
    // 0. Clear stale intermediate activations so emplace succeeds each run
    const Graph& g = model_->graph();
    for (const Node& node : g.nodes)
        for (const auto& out_name : node.outputs)
            value_map_.erase(out_name);

    // 1. Upload inputs to GPU and insert into value map
    for (auto& [name, tensor] : inputs) {
        tensor.cuda();
        value_map_[name] = std::move(tensor);
    }

    // 2. Dispatch each node in topological order
    for (const Node& node : g.nodes) {
        auto kernel = internal::OpRegistry::instance().create(node.op_type);
        internal::KernelContext ctx{node, value_map_, stream_};
        kernel->execute(ctx, value_map_);
    }

    // 3. Sync stream before reading results
    cudaStreamSynchronize(stream_);

    // 4. Collect and return outputs — erase from value_map_ so emplace
    //    succeeds on the next run() call (move leaves a zombie key otherwise)
    std::unordered_map<std::string, Tensor> outputs;
    for (const auto& name : g.output_names) {
        outputs[name] = std::move(value_map_.at(name));
        value_map_.erase(name);
    }
    return outputs;
}

// ---------------------------------------------------------------------------
// CPUExecutor::run
// ---------------------------------------------------------------------------

std::unordered_map<std::string, Tensor>
CPUExecutor::run(std::unordered_map<std::string, Tensor> inputs) {
    // 0. Clear stale intermediate activations
    const Graph& g = model_->graph();
    for (const Node& node : g.nodes)
        for (const auto& out_name : node.outputs)
            value_map_.erase(out_name);

    // 1. Insert inputs (CPU tensors, no device transfer needed)
    for (auto& [name, tensor] : inputs)
        value_map_[name] = std::move(tensor);

    // 2. Dispatch each node — ops dispatch to CPUStorage via tensor's storage ptr
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
