#include "tensor/tensor.h"
#include "tensor/tensor_functions.h"
#include <gtest/gtest.h>

namespace {

class TensorTest : public ::testing::Test {
protected:
    const std::vector<int64_t> shape_{8192, 8192};
};

// Allocate a tensor (defaults to CPU) and call cpu() — should be a no-op
TEST_F(TensorTest, AllocateAndMoveToCPU) {
    tinyinfer::Tensor t(shape_);
    t.cpu();  // already on CPU, should be a no-op
    EXPECT_FALSE(t.empty());
    EXPECT_NE(t.data_ptr(), nullptr);
}

// Add two tensors on CPU
TEST_F(TensorTest, AddTensorsOnCPU) {
    tinyinfer::Tensor a = tinyinfer::zeros(shape_);
    a.fill(3.0f);

    tinyinfer::Tensor b = tinyinfer::zeros(shape_);
    b.fill(4.0f);

    tinyinfer::Tensor c = a + b;
    EXPECT_FLOAT_EQ(c.at({0, 0}), 7.0f);
    EXPECT_FLOAT_EQ(c.at({1, 3}), 7.0f);
}

// Add two tensors on GPU, bring result back to CPU to verify
TEST_F(TensorTest, AddTensorsOnGPU) {
    tinyinfer::Tensor a = tinyinfer::zeros(shape_);
    a.fill(3.0f);
    a.cuda();

    tinyinfer::Tensor b = tinyinfer::zeros(shape_);
    b.fill(4.0f);
    b.cuda();

    // Loop to keep GPU busy long enough for nvtop to show activity
    tinyinfer::Tensor c = a + b;
    for (int i = 0; i < 500; ++i) {
        c = a + b;
    }
    EXPECT_FLOAT_EQ(c.at({0, 0}), 7.0f);
    EXPECT_FLOAT_EQ(c.at({1, 3}), 7.0f);
}

// Adding tensors on different devices should throw
TEST_F(TensorTest, AddTensorsMismatchedDevices) {
    tinyinfer::Tensor a = tinyinfer::zeros(shape_);
    a.fill(1.0f);

    tinyinfer::Tensor b = tinyinfer::zeros(shape_);
    b.fill(1.0f);
    b.cuda();

    EXPECT_THROW(a + b, std::runtime_error);
}

}  // namespace
