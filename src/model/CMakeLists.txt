cmake_minimum_required(VERSION 3.17)

find_package(Protobuf REQUIRED)

# Always generate proto locally so add_library can find the GENERATED file
protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnx/onnx.proto
)

add_library(tinyinfer_model STATIC
    ${ONNX_PROTO_SRCS}
    model.cpp
    onnx_parser.cpp
)

target_include_directories(tinyinfer_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}     # for generated onnx.pb.h
    ${Protobuf_INCLUDE_DIRS}
)

target_include_directories(tinyinfer_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}     # for onnx_parser.h (internal)
)

target_link_libraries(tinyinfer_model PUBLIC
    tinyinfer_graph
    ${Protobuf_LIBRARIES}
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
