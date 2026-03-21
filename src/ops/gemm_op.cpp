#include "ops/op_registry.h"

namespace tinyinfer {

class GemmOp : public OpKernel {
public:
    void execute(const KernelContext& ctx,
                 std::unordered_map<std::string, Tensor>& value_map) override {
        const Node& node = ctx.node;
        const Tensor& input  = value_map.at(node.inputs[0]);
        const Tensor& weight = value_map.at(node.inputs[1]);
        const Tensor& bias   = value_map.at(node.inputs[2]);
        value_map.emplace(node.outputs[0], input.gemm(weight, bias));
    }
};

static OpRegistrar gemm_registrar(OpType::Gemm, []() -> std::unique_ptr<OpKernel> {
    return std::make_unique<GemmOp>();
});

}  // namespace tinyinfer
