cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_library(tinyinfer_ops INTERFACE)

target_include_directories(tinyinfer_ops INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(tinyinfer_ops INTERFACE
    tinyinfer_graph
    tinyinfer_tensor
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
