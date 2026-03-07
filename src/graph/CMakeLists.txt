cmake_minimum_required(VERSION 3.17)

add_library(tinyinfer_graph STATIC
    # graph.cpp  # TODO
)

target_include_directories(tinyinfer_graph PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(tinyinfer_graph PUBLIC
    tinyinfer_tensor
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
