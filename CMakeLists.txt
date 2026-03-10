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
# Modules (dependency order)
# --------------------------------------------------------------------------
include(CTest)

add_subdirectory(src/tensor)
add_subdirectory(src/graph)
add_subdirectory(src/model)
add_subdirectory(src/ops)
add_subdirectory(src/runtime)
add_subdirectory(src/data)
add_subdirectory(src/eval)

# --------------------------------------------------------------------------
# Demo executable
# --------------------------------------------------------------------------
add_executable(tinyinfer_demo src/main.cpp)
target_link_libraries(tinyinfer_demo PRIVATE
    tinyinfer_eval
    tinyinfer_runtime
    tinyinfer_model
    tinyinfer_ops
    tinyinfer_graph
    tinyinfer_tensor
    tinyinfer_data
)
