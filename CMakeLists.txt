cmake_minimum_required(VERSION 3.12)
project(
  kamping
  DESCRIPTION "An MPI wrapper which makes using MPI feel like C++"
  LANGUAGES CXX
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # folder support for IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # this has to be enabled in the main CMakeLists file
  include(CTest)

  # find Doxygen
  find_package(Doxygen)
  if(Doxygen_FOUND)
    #add_subdirectory(docs)
  endif()
else()
  message(STATUS "Doxygen not found, not building docs")
endif()

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

option(KAMPING_WARNINGS_ARE_ERRORS OFF)
option(KAMPING_BUILD_EXAMPLES_AND_TESTS OFF)

set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
find_package(MPI REQUIRED)

add_subdirectory(extern)

add_library(kamping INTERFACE)
target_include_directories(kamping INTERFACE include)

# set C++ standard to C++17
target_compile_features(kamping INTERFACE cxx_std_17)
target_link_libraries(kamping INTERFACE MPI::MPI_CXX)

list(
  APPEND KAMPING_WARNING_FLAGS
  "-Wall"
  "-Wextra"
  "-Wconversion"
  "-Wnon-virtual-dtor"
  "-Woverloaded-virtual"
  "-Wshadow"
  "-Wsign-conversion"
  "-Wundef"
  "-Wunreachable-code"
  "-Wunused"
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(
    APPEND KAMPING_WARNING_FLAGS
    "-Wcast-align"
    "-Wnull-dereference"
    "-Wpedantic"
  )
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(
    APPEND KAMPING_WARNING_FLAGS
    "-Wcast-align"
    "-Wnull-dereference"
    "-Wpedantic"
    "-Wnoexcept"
    "-Wsuggest-attribute=const"
    "-Wsuggest-attribute=noreturn"
    "-Wsuggest-override"
  )
endif()

# OFF by default.
message(STATUS "${KAMPING_WARNINGS_ARE_ERRORS}")
if(KAMPING_WARNINGS_ARE_ERRORS)
  list(
    APPEND KAMPING_WARNING_FLAGS
    "-Werror"
  )
endif()

add_library(kamping::kamping ALIAS kamping)

# Testing and examples are only built if this is the main project
# or of KAMPING_BUILD_EXAMPLES_AND_TESTS is set (OFF by default)
if(
  CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR KAMPING_BUILD_EXAMPLES_AND_TESTS
)
  add_subdirectory(examples)
  add_subdirectory(tests)
endif()
