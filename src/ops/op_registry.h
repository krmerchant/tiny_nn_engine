#pragma once
#include <functional>
#include <unordered_map>
#include <memory>
#include <string>
#include "graph/graph.h"
#include "tensor/tensor.h"

namespace tinyinfer::internal {

// Context passed to each kernel at inference time
struct KernelContext {
    const Node& node;
    const std::unordered_map<std::string, Tensor>& value_map;  // name → tensor (activations + weights)
    cudaStream_t stream;
};

// Abstract base for all op implementations
class OpKernel {
public:
    virtual ~OpKernel() = default;

    // Execute the op; write outputs into value_map via non-const ref
    virtual void execute(const KernelContext& ctx,
                         std::unordered_map<std::string, Tensor>& value_map) = 0;
};

// Singleton factory: OpType → OpKernel factory
class OpRegistry {
public:
    using Factory = std::function<std::unique_ptr<OpKernel>()>;

    static OpRegistry& instance();

    void register_op(OpType type, Factory factory);
    std::unique_ptr<OpKernel> create(OpType type) const;

private:
    OpRegistry() = default;
    std::unordered_map<int, Factory> factories_;
};

// Helper for self-registration at static-init time
struct OpRegistrar {
    OpRegistrar(OpType type, OpRegistry::Factory factory) {
        OpRegistry::instance().register_op(type, std::move(factory));
    }
};

}  // namespace tinyinfer::internal
