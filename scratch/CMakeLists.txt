cmake_minimum_required(VERSION 3.10)
project(print_onnx CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(Protobuf REQUIRED)

# Generate C++ bindings from onnx.proto
set(ONNX_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnx/onnx.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${ONNX_PROTO})

## print_onnx
add_executable(print_onnx print_onnx.cpp ${PROTO_SRCS})

target_include_directories(print_onnx PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS}
)
target_link_libraries(print_onnx PRIVATE ${Protobuf_LIBRARIES})


## model_onnx_playground
add_executable(model_onnx_playground model_onnx_playground.cpp ${PROTO_SRCS})
target_include_directories(model_onnx_playground PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS}
)
target_link_libraries(model_onnx_playground PRIVATE ${Protobuf_LIBRARIES})

