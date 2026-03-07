cmake_minimum_required(VERSION 3.17)

add_library(tinyinfer_eval STATIC
    # evaluator.cpp  # TODO
)

target_include_directories(tinyinfer_eval PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(tinyinfer_eval PUBLIC
    tinyinfer_data
    tinyinfer_runtime
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
