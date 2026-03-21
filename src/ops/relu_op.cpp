#include "ops/op_registry.h"

namespace tinyinfer::internal {

class ReluOp : public OpKernel {
public:
    void execute(const KernelContext& ctx,
                 std::unordered_map<std::string, Tensor>& value_map) override {
        const Node& node = ctx.node;
        const Tensor& input = value_map.at(node.inputs[0]);
        value_map.emplace(node.outputs[0], input.relu(ctx.stream));
    }
};

static OpRegistrar relu_registrar(OpType::Relu, []() -> std::unique_ptr<OpKernel> {
    return std::make_unique<ReluOp>();
});

}  // namespace tinyinfer::internal
