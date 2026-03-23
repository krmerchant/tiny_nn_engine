cmake_minimum_required(VERSION 3.17)

find_package(CUDAToolkit REQUIRED)

add_library(tinyinfer_runtime STATIC executor.cpp profiler.cpp)

target_include_directories(tinyinfer_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(tinyinfer_runtime PUBLIC
    tinyinfer_model
    tinyinfer_ops
    CUDA::cudart
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
