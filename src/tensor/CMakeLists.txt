cmake_minimum_required(VERSION 3.17)

find_package(CUDAToolkit REQUIRED)

add_library(tinyinfer_tensor STATIC
    tensor.cpp
)

target_include_directories(tinyinfer_tensor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..   # allows: #include "tensor/tensor.h"
)

target_link_libraries(tinyinfer_tensor PUBLIC
    CUDA::cudart
    CUDA::cublas
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
