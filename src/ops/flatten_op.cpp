#include "ops/op_registry.h"

namespace tinyinfer {

class FlattenOp : public OpKernel {
public:
    void execute(const KernelContext& ctx,
                 std::unordered_map<std::string, Tensor>& value_map) override {
        const Node& node = ctx.node;
        // ONNX Flatten: axis attribute splits shape into outer and inner product.
        // Default axis=1: outer = shape[0], inner = product(shape[1..]).
        Tensor& input = value_map.at(node.inputs[0]);
        const auto& shape = input.shape();
        int64_t axis = 1;
        auto it = node.int_attrs.find("axis");
        if (it != node.int_attrs.end()) axis = it->second;

        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < axis; ++i)          outer *= shape[i];
        for (int64_t i = axis; i < (int64_t)shape.size(); ++i) inner *= shape[i];

        input.reshape_({outer, inner});
        // Flatten is a view — move/alias into outputs[0]; no data copy needed.
        // Insert a reference by moving the existing entry to the output key.
        value_map.emplace(node.outputs[0], std::move(value_map.extract(node.inputs[0]).mapped()));
    }
};

static OpRegistrar flatten_registrar(OpType::Flatten, []() -> std::unique_ptr<OpKernel> {
    return std::make_unique<FlattenOp>();
});

}  // namespace tinyinfer
