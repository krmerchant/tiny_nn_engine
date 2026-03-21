#include "model/model.h"
#include <gtest/gtest.h>

TEST(ModelTest, LinearLayerInitializerValues) {
    auto model = tinyinfer::Model::load(TEST_LINEAR_ONNX_PATH);
    const auto& graph = model.graph();

    // One Gemm node
    ASSERT_EQ(graph.nodes.size(), 1u);
    EXPECT_EQ(graph.nodes[0].op_type, tinyinfer::OpType::Gemm);

    // Weight: shape [1, 3], values [1.0, 2.0, 3.0]
    ASSERT_EQ(graph.initializers.count("fc.weight"), 1u);
    const auto& w = graph.initializers.at("fc.weight");
    ASSERT_EQ(w.shape(), (std::vector<int64_t>{1, 3}));
    EXPECT_FLOAT_EQ(w.at({0u, 0u}), 1.0f);
    EXPECT_FLOAT_EQ(w.at({0u, 1u}), 2.0f);
    EXPECT_FLOAT_EQ(w.at({0u, 2u}), 3.0f);

    // Bias: shape [1], value [0.5]
    ASSERT_EQ(graph.initializers.count("fc.bias"), 1u);
    const auto& b = graph.initializers.at("fc.bias");
    ASSERT_EQ(b.shape(), (std::vector<int64_t>{1}));
    EXPECT_FLOAT_EQ(b.at({0u}), 0.5f);
}
