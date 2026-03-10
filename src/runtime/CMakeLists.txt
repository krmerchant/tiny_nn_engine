cmake_minimum_required(VERSION 3.17)

find_package(CUDAToolkit REQUIRED)

add_library(tinyinfer_runtime INTERFACE)

target_include_directories(tinyinfer_runtime INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(tinyinfer_runtime INTERFACE
    tinyinfer_model
    tinyinfer_ops
    CUDA::cudart
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
