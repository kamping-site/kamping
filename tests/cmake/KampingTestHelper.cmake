include(KaTestrophe)
include(GoogleTest)

# Convenience wrapper for adding tests for KaMPI.ng
# this creates the target, links googletest and kamping, enables warnings and registers the test
#
# TARGET_NAME the target name
# FILES the files of the target
#
# example: kamping_register_test(mytarget FILES mytarget.cpp)
function(kamping_register_test KAMPING_TARGET_NAME)
  cmake_parse_arguments(
    "KAMPING"
    "NO_EXCEPTION_MODE"
    ""
    "FILES"
    ${ARGN}
    )
  add_executable(${KAMPING_TARGET_NAME} ${KAMPING_FILES})
  target_link_libraries(${KAMPING_TARGET_NAME} PRIVATE gtest gtest_main gmock kamping)
  target_compile_options(${KAMPING_TARGET_NAME} PRIVATE ${KAMPING_WARNING_FLAGS})
  gtest_discover_tests(${KAMPING_TARGET_NAME} WORKING_DIRECTORY ${PROJECT_DIR})
  if (NOT KAMPING_NO_EXCEPTION_MODE)
    target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -DKAMPING_EXCEPTION_MODE)
  endif ()
endfunction()

# Convenience wrapper for adding tests for KaMPI.ng which rely on MPI
# this creates the target, links googletest, kamping and MPI, enables warnings and registers the tests
#
# TARGET_NAME the target name
# FILES the files of the target
# CORES the number of MPI ranks to run the test for
#
# example: kamping_register_mpi_test(mytarget FILES mytarget.cpp CORES 1 2 4 8)
function(kamping_register_mpi_test KAMPING_TARGET_NAME)
  cmake_parse_arguments(
    "KAMPING"
    "NO_EXCEPTION_MODE"
    ""
    "FILES;CORES"
    ${ARGN}
    )
  katestrophe_add_test_executable(${KAMPING_TARGET_NAME} FILES ${KAMPING_FILES})
  target_link_libraries(${KAMPING_TARGET_NAME} PRIVATE kamping)
  katestrophe_add_mpi_test(${KAMPING_TARGET_NAME} CORES ${KAMPING_CORES} DISCOVER_TESTS)
  if (NOT KAMPING_NO_EXCEPTION_MODE)
    target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -DKAMPING_EXCEPTION_MODE)
  endif ()
endfunction()

# Convenience wrapper for registering a set of tests that should fail to compile and require KaMPI.ng to be linked.
#
# TARGET prefix for the targets to be built
# FILES the list of files to include in the target
# SECTIONS sections of the compilation test to build
#
function(kamping_register_compilation_failure_test KAMPING_TARGET_NAME)
  cmake_parse_arguments(
    "KAMPING"
    "NO_EXCEPTION_MODE"
    ""
    "FILES;SECTIONS"
    ${ARGN}
    )
  katestrophe_add_compilation_failure_test(
    TARGET ${KAMPING_TARGET_NAME}
    FILES ${KAMPING_FILES}
    SECTIONS ${KAMPING_SECTIONS}
    LIBRARIES kamping
    )
  if (NOT KAMPING_NO_EXCEPTION_MODE)
    target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -DKAMPING_EXCEPTION_MODE)
  endif ()
endfunction()
