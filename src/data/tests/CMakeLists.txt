add_executable(test_data test_data.cpp)
target_link_libraries(test_data PRIVATE tinyinfer_data)
add_test(NAME test_data COMMAND test_data)
