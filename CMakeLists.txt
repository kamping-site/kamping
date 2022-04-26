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
  if(DOXYGEN_FOUND)
    if (DOXYGEN_VERSION VERSION_LESS "1.9.2")
      message(WARNING "Doxygen must be version 1.9.2 or newer. Documentation may not be displayed correctly and CI may not pass even if checks pass locally.")
    endif()
    add_custom_target(docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/docs/Doxyfile
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMENT "Generating Documentation"
      VERBATIM)
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
option(KAMPING_TESTS_DISCOVER OFF)

set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
find_package(MPI REQUIRED)

add_subdirectory(extern)

add_library(kamping_base INTERFACE)
target_include_directories(kamping_base INTERFACE include)

# set C++ standard to C++17
target_compile_features(kamping_base INTERFACE cxx_std_17)
target_link_libraries(kamping_base INTERFACE MPI::MPI_CXX)

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
    "-Wextra-semi"
    "-Wno-gnu-zero-variadic-macro-arguments"
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
if(KAMPING_WARNINGS_ARE_ERRORS)
  list(
    APPEND KAMPING_WARNING_FLAGS
    "-Werror"
  )
endif()

# Target for user-code
add_library(kamping INTERFACE)
target_link_libraries(kamping INTERFACE kamping_base)

# If enabled, use exceptions, otherwise use KASSERT()
option(KAMPING_EXCEPTION_MODE "Use exceptions to report recoverable errors." ON)
if (KAMPING_EXCEPTION_MODE)
  target_compile_definitions(kamping INTERFACE -DKAMPING_EXCEPTION_MODE)
  message(STATUS "Build with exceptions enabled.")
else()
  message(STATUS "Build with exceptions disabled. Assertions are used instead.")
endif()

# The assertion level controls which assertions are enabled during runtime:
#
# Level 0: Disable all assertions
# Level 1: Exception assertions = only enable exceptions (if not in exception mode)
# Level 2: Light assertions = assertions that do not affect the running time of library operations significantly.
# Level 3: Normal assertions = assertions that might slow down some operations of the library by a constant factor.
#          Should only be used in debug mode.
# Level 4: Light communication assertions = assertions that perform additional communication causing small running
#          time overheads.
# Level 5: Heavy communication assertions = assertions that perform additional communication causing significant
#          running time overheads.
# Level 6: Heavy assertions = assertions that introduce overhead which renders some library operations infeasible when
#          invoked with any significant work load.
#
# Assertion levels can be set explicitly using the -DKAMPING_ASSERTION_LEVEL=... flag. If no level is set explicitly,
# we set it to 1 (exceptions only) in Release mode and 3 (up to normal assertions) in Debug mode.
set(KASSERT_ASSERTION_LEVEL $<IF:$<CONFIG:Debug>,30,10> CACHE STRING "Higher values enable more complex assertions. To disable all runtime checks, set to '0'.")
set_property(CACHE KASSERT_ASSERTION_LEVEL PROPERTY STRINGS 0 10 20 30 40 50 60)

# Constant for the highest assertion level used by KaMPI.ng
set(KASSERT_HIGHEST_ASSERTION_LEVEL 60)

add_subdirectory(extern/kassert)
target_include_directories(kamping_base INTERFACE extern/kassert/include)
target_link_libraries(kamping_base INTERFACE kassert)


add_library(kamping::kamping ALIAS kamping)

# Testing and examples are only built if this is the main project
# or if KAMPING_BUILD_EXAMPLES_AND_TESTS is set (OFF by default)
if(
  CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR KAMPING_BUILD_EXAMPLES_AND_TESTS
)
  add_subdirectory(examples)
  add_subdirectory(tests)
endif()
