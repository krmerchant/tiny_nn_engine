cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_library(tinyinfer_ops STATIC
    # op_registry.cpp    # TODO
    # gemm_op.cpp        # TODO
    # gemm_kernel.cu     # TODO
    # relu_op.cpp        # TODO
    # relu_kernel.cu     # TODO
    # softmax_op.cpp     # TODO
    # softmax_kernel.cu  # TODO
)

target_include_directories(tinyinfer_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(tinyinfer_ops PUBLIC
    tinyinfer_graph
    tinyinfer_tensor
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
