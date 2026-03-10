cmake_minimum_required(VERSION 3.17)

add_library(tinyinfer_data INTERFACE)

target_include_directories(tinyinfer_data INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
