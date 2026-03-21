#include "runtime/executor.h"
#include "ops/op_registry.h"

namespace tinyinfer {

Executor::~Executor() {
    if (stream_) cudaStreamDestroy(stream_);
}

std::unordered_map<std::string, Tensor>
Executor::run(std::unordered_map<std::string, Tensor> inputs) {
    // 1. Upload inputs to GPU and insert into value map
    for (auto& [name, tensor] : inputs) {
        tensor.cuda();
        value_map_[name] = std::move(tensor);
    }

    // 2. Dispatch each node in topological order
    const Graph& g = model_->graph();
    for (const Node& node : g.nodes) {
        auto kernel = internal::OpRegistry::instance().create(node.op_type);
        internal::KernelContext ctx{node, value_map_, stream_};
        kernel->execute(ctx, value_map_);
    }

    // 3. Sync stream before reading results
    cudaStreamSynchronize(stream_);

    // 4. Collect and return outputs (move out; they'll be recomputed on next run)
    std::unordered_map<std::string, Tensor> outputs;
    for (const auto& name : g.output_names)
        outputs[name] = std::move(value_map_.at(name));
    return outputs;
}

}  // namespace tinyinfer
