add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE tinyinfer_runtime)
add_test(NAME test_runtime COMMAND test_runtime)
