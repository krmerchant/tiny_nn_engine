add_executable(test_model test_model.cpp)
target_link_libraries(test_model PRIVATE tinyinfer_model)
add_test(NAME test_model COMMAND test_model)
