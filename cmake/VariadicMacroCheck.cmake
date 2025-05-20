# CheckVariadicMacroSupport.cmake
# --------------------------------
# Detects support for __VA_OPT__ and , ##__VA_ARGS__ in the compiler.

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

# Check for __VA_OPT__ support (requires C++20)
set(_VA_OPT_TEST_SRC [[
#include <cstdio>
#define LOG(fmt, ...) std::printf("[LOG] " fmt __VA_OPT__(, __VA_ARGS__))
int main() {
    LOG("Hello\\n");
    LOG("value=%d\\n", 42);
    return 0;
}
]])

cmake_push_check_state()
set(CMAKE_REQUIRED_FLAGS "-std=c++11")
check_cxx_source_compiles("${_VA_OPT_TEST_SRC}" VARIADIC_MACRO_USE_VA_OPT)
cmake_pop_check_state()


cmake_push_check_state()
set(CMAKE_REQUIRED_FLAGS "-std=c++20")
check_cxx_source_compiles("${_VA_OPT_TEST_SRC}" VARIADIC_MACRO_REQUIRES_20)
cmake_pop_check_state()

# # Determine the required C++ standard
# if(VARIADIC_MACRO_USE_VA_OPT)
#   set(VARIADIC_MACRO_REQUIRED_STANDARD "cxx_std_20")
# elseif(VARIADIC_MACRO_USE_GNU)
#   set(VARIADIC_MACRO_REQUIRED_STANDARD "gnu++17")
# else()
#   set(VARIADIC_MACRO_REQUIRED_STANDARD "cxx_std_17")
# endif()
