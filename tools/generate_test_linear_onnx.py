"""
Generate a minimal test_linear.onnx for use by test_model unit tests.

Graph: input(1x3) --[Gemm]--> output(1x1)
  fc.weight = [[1.0, 2.0, 3.0]]  shape [1, 3]
  fc.bias   = [0.5]               shape [1]

Usage:
    python3 tools/generate_test_linear_onnx.py
Writes: src/model/tests/test_linear.onnx
"""

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import os

WEIGHT_VALUES = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # shape [1, 3]
BIAS_VALUES   = np.array([0.5],              dtype=np.float32)  # shape [1]

W_init = numpy_helper.from_array(WEIGHT_VALUES, name="fc.weight")
b_init = numpy_helper.from_array(BIAS_VALUES,   name="fc.bias")

gemm_node = helper.make_node(
    "Gemm",
    inputs=["input", "fc.weight", "fc.bias"],
    outputs=["output"],
    name="gemm0",
    transB=1,
)

graph = helper.make_graph(
    [gemm_node],
    "test_linear",
    [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, 3])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
    initializer=[W_init, b_init],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
onnx.checker.check_model(model)

out_path = os.path.join(
    os.path.dirname(__file__), "..", "src", "model", "tests", "test_linear.onnx"
)
out_path = os.path.normpath(out_path)
onnx.save(model, out_path)
print(f"Wrote {out_path}")
print(f"  fc.weight = {WEIGHT_VALUES.tolist()}")
print(f"  fc.bias   = {BIAS_VALUES.tolist()}")
