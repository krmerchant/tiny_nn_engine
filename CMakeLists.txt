cmake_minimum_required(VERSION 3.17)
project(tinyinfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CUDA 10.1 (nvcc) supports up to C++14 for device code
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GTX 1650 → sm_75
set(CMAKE_CUDA_ARCHITECTURES 75)

# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(Protobuf REQUIRED)

# --------------------------------------------------------------------------
# Generate ONNX protobuf bindings
# --------------------------------------------------------------------------
protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
    third_party/onnx/onnx.proto
)

# --------------------------------------------------------------------------
# tinyinfer static library
# --------------------------------------------------------------------------
add_library(tinyinfer STATIC
    ${ONNX_PROTO_SRCS}
    src/tensor.cpp
    src/graph.cpp
    src/onnx_parser.cpp
    src/model.cpp
    src/op_registry.cpp
    src/executor.cpp
    src/profiler.cpp
    src/mnist_loader.cpp
    src/evaluator.cpp
    src/ops/gemm_op.cpp
    src/ops/relu_op.cpp
    src/ops/softmax_op.cpp
    src/ops/gemm_kernel.cu
    src/ops/relu_kernel.cu
    src/ops/softmax_kernel.cu
)

target_include_directories(tinyinfer PUBLIC
    include
    ${CMAKE_CURRENT_BINARY_DIR}   # for generated onnx.pb.h
    ${Protobuf_INCLUDE_DIRS}
)
target_include_directories(tinyinfer PRIVATE
    src   # for internal onnx_parser.h
)

target_link_libraries(tinyinfer PUBLIC
    ${Protobuf_LIBRARIES}
    CUDA::cudart
)

# Make sure CUDA sources see C++ standard headers
set_target_properties(tinyinfer PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# --------------------------------------------------------------------------
# tinyinfer_demo executable
# --------------------------------------------------------------------------
add_executable(tinyinfer_demo src/main.cpp)
target_link_libraries(tinyinfer_demo PRIVATE tinyinfer)
