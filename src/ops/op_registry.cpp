#include "ops/op_registry.h"
#include <stdexcept>

namespace tinyinfer {

OpRegistry& OpRegistry::instance() {
    static OpRegistry reg;
    return reg;
}

void OpRegistry::register_op(OpType type, Factory f) {
    factories_[static_cast<int>(type)] = std::move(f);
}

std::unique_ptr<OpKernel> OpRegistry::create(OpType type) const {
    auto it = factories_.find(static_cast<int>(type));
    if (it == factories_.end())
        throw std::runtime_error("OpRegistry: unknown op type");
    return it->second();
}

}  // namespace tinyinfer
