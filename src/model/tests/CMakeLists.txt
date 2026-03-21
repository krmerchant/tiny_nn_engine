add_executable(test_model test_model.cpp)
target_compile_definitions(test_model PRIVATE
    TEST_LINEAR_ONNX_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test_linear.onnx"
)
target_link_libraries(test_model PRIVATE tinyinfer_model GTest::gtest_main)
add_test(NAME test_model COMMAND test_model)
