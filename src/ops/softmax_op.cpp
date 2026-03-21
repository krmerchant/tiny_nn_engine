#include "ops/op_registry.h"

namespace tinyinfer {

class SoftmaxOp : public OpKernel {
public:
    void execute(const KernelContext& ctx,
                 std::unordered_map<std::string, Tensor>& value_map) override {
        const Node& node = ctx.node;
        const Tensor& input = value_map.at(node.inputs[0]);
        value_map.emplace(node.outputs[0], input.softmax(ctx.stream));
    }
};

static OpRegistrar softmax_registrar(OpType::Softmax, []() -> std::unique_ptr<OpKernel> {
    return std::make_unique<SoftmaxOp>();
});

}  // namespace tinyinfer
