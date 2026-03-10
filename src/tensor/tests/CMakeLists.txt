include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE tinyinfer_tensor GTest::gtest_main)
add_test(NAME test_tensor COMMAND test_tensor)
