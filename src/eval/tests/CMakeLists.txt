add_executable(test_eval test_eval.cpp)
target_link_libraries(test_eval PRIVATE tinyinfer_eval)
add_test(NAME test_eval COMMAND test_eval)
