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

// gemm: input(1,3) * weight(2,3)^T + bias(2,) = [1,2,3]*[[1,0,0],[0,1,0]]^T + [10,20]
//      = [1, 2] + [10, 20] = [11, 22]
TEST_F(TensorTest, GemmCorrectness) {
    tinyinfer::Tensor input({1.f, 2.f, 3.f}, {1, 3});
    tinyinfer::Tensor weight({1.f, 0.f, 0.f, 0.f, 1.f, 0.f}, {2, 3});
    tinyinfer::Tensor bias({10.f, 20.f}, {2});
    input.cuda(); weight.cuda(); bias.cuda();

    tinyinfer::Tensor out = input.gemm(weight, bias);

    out.cpu();
    EXPECT_FLOAT_EQ(out.at({0, 0}), 11.f);
    EXPECT_FLOAT_EQ(out.at({0, 1}), 22.f);
}

// relu: [-1, 0, 2, -3] -> [0, 0, 2, 0]
TEST_F(TensorTest, ReluCorrectness) {
    tinyinfer::Tensor t({-1.f, 0.f, 2.f, -3.f}, {1, 4});
    t.cuda();
    tinyinfer::Tensor out = t.relu();
    out.cpu();
    EXPECT_FLOAT_EQ(out.at({0, 0}), 0.f);
    EXPECT_FLOAT_EQ(out.at({0, 1}), 0.f);
    EXPECT_FLOAT_EQ(out.at({0, 2}), 2.f);
    EXPECT_FLOAT_EQ(out.at({0, 3}), 0.f);
}

}  // namespace
