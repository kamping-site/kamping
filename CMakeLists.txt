cmake_minimum_required(VERSION 3.12)
project(kamping
  DESCRIPTION "An MPI wrapper which makes using MPI feel like C++"
  LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # this has to be enabled in the main CMakeLists file
  include(CTest)

  # find Doxygen
  find_package(Doxygen)
  if(Doxygen_FOUND)
    #add_subdirectory(doc)
    endif()
  else()
    message(STATUS "Doxygen not found, not building docs")
endif()

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

find_package(MPI REQUIRED)

add_subdirectory(extern)

add_library(kamping INTERFACE)
target_include_directories(kamping INTERFACE include)

# set C++ standard to C++17
target_compile_features(kamping INTERFACE cxx_std_17)
target_link_libraries(kamping INTERFACE MPI::MPI_CXX)

set(KAMPING_WARNING_FLAGS $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Intel>>:
  -Wall -Wextra -pedantic -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:
  /WX /W4>)

add_library(kamping::kamping ALIAS kamping)

# Testing only available if this is the main app
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    AND BUILD_TESTING)
  add_subdirectory(examples)
  add_subdirectory(tests)
endif()
