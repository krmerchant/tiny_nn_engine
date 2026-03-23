cmake_minimum_required(VERSION 3.17)

add_library(tinyinfer_data STATIC
    mnist_loader.cpp
)

target_include_directories(tinyinfer_data PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(tinyinfer_data PUBLIC
    tinyinfer_tensor
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
