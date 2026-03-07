add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops PRIVATE tinyinfer_ops)
add_test(NAME test_ops COMMAND test_ops)
