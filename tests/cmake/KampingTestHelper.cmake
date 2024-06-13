include(KaTestrophe)
include(GoogleTest)

function (kamping_set_kassert_flags KAMPING_TARGET_NAME)
    cmake_parse_arguments("KAMPING" "NO_EXCEPTION_MODE" "" "" ${ARGN})

    # Use global assertion level
    target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -DKASSERT_ASSERTION_LEVEL=${KASSERT_ASSERTION_LEVEL})

    # Explicitly specify exception mode for tests, default to no exception mode
    if (NOT KAMPING_NO_EXCEPTION_MODE)
        target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -DKASSERT_EXCEPTION_MODE)
    endif ()
endfunction ()

# Convenience wrapper for adding tests for KaMPIng this creates the target, links googletest and kamping, enables
# warnings and registers the test
#
# TARGET_NAME the target name FILES the files of the target
#
# example: kamping_register_test(mytarget FILES mytarget.cpp)
function (kamping_register_test KAMPING_TARGET_NAME)
    cmake_parse_arguments("KAMPING" "NO_GLIBCXX_DEBUG_CONTAINERS" "" "FILES" ${ARGN})
    add_executable(${KAMPING_TARGET_NAME} ${KAMPING_FILES})
    target_link_libraries(${KAMPING_TARGET_NAME} PRIVATE gtest gtest_main gmock kamping_base)
    target_compile_options(${KAMPING_TARGET_NAME} PRIVATE ${KAMPING_WARNING_FLAGS})
    gtest_discover_tests(${KAMPING_TARGET_NAME} WORKING_DIRECTORY ${PROJECT_DIR} DISCOVERY_MODE PRE_TEST)
    kamping_set_kassert_flags(${KAMPING_TARGET_NAME} ${ARGN})
    if (NOT ${KAMPING_NO_GLIBCXX_DEBUG_CONTAINERS})
        target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -D_GLIBCXX_DEBUG)
        target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -D_GLIBCXX_DEBUG_PEDANTIC)
    endif ()

    if (KAMPING_TEST_ENABLE_SANITIZERS)
        target_compile_options(${KAMPING_TARGET_NAME} PRIVATE -fsanitize=address)
        target_link_options(${KAMPING_TARGET_NAME} PRIVATE -fsanitize=address)
        target_compile_options(${KAMPING_TARGET_NAME} PRIVATE -fsanitize=undefined)
        target_link_options(${KAMPING_TARGET_NAME} PRIVATE -fsanitize=undefined)
        target_compile_options(${KAMPING_TARGET_NAME} PRIVATE -fno-sanitize-recover=all)
        target_link_options(${KAMPING_TARGET_NAME} PRIVATE -fno-sanitize-recover=all)
    endif ()
endfunction ()

# Convenience wrapper for adding tests for KaMPIng which rely on MPI this creates the target, links googletest, kamping
# and MPI, enables warnings and registers the tests
#
# TARGET_NAME the target name FILES the files of the target CORES the number of MPI ranks to run the test for
#
# example: kamping_register_mpi_test(mytarget FILES mytarget.cpp CORES 1 2 4 8)
function (kamping_register_mpi_test KAMPING_TARGET_NAME)
    cmake_parse_arguments("KAMPING" "NO_GLIBCXX_DEBUG_CONTAINERS" "" "FILES;CORES" ${ARGN})
    katestrophe_add_test_executable(${KAMPING_TARGET_NAME} FILES ${KAMPING_FILES})
    target_link_libraries(${KAMPING_TARGET_NAME} PRIVATE kamping_base)
    if (KAMPING_TESTS_DISCOVER)
        katestrophe_add_mpi_test(
            ${KAMPING_TARGET_NAME}
            CORES ${KAMPING_CORES}
            DISCOVER_TESTS
        )
    else ()
        katestrophe_add_mpi_test(${KAMPING_TARGET_NAME} CORES ${KAMPING_CORES})
    endif ()
    kamping_set_kassert_flags(${KAMPING_TARGET_NAME} ${ARGN})
    if (NOT ${KAMPING_NO_GLIBCXX_DEBUG_CONTAINERS})
        target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -D_GLIBCXX_DEBUG)
        target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -D_GLIBCXX_DEBUG_PEDANTIC)
    endif ()
    if (KAMPING_TEST_ENABLE_SANITIZERS)
        target_compile_options(${KAMPING_TARGET_NAME} PRIVATE -fsanitize=undefined)
        target_link_options(${KAMPING_TARGET_NAME} PRIVATE -fsanitize=undefined)
        target_compile_options(${KAMPING_TARGET_NAME} PRIVATE -fsanitize=address)
        target_link_options(${KAMPING_TARGET_NAME} PRIVATE -fsanitize=address)
        target_compile_options(${KAMPING_TARGET_NAME} PRIVATE -fno-sanitize-recover=all)
        target_link_options(${KAMPING_TARGET_NAME} PRIVATE -fno-sanitize-recover=all)
    endif ()
endfunction ()

# Convenience wrapper for registering a set of tests that should fail to compile and require KaMPIng to be linked.
#
# TARGET prefix for the targets to be built FILES the list of files to include in the target SECTIONS sections of the
# compilation test to build
#
function (kamping_register_compilation_failure_test KAMPING_TARGET_NAME)
    cmake_parse_arguments("KAMPING" "NO_EXCEPTION_MODE" "" "FILES;SECTIONS" ${ARGN})
    katestrophe_add_compilation_failure_test(
        TARGET ${KAMPING_TARGET_NAME}
        FILES ${KAMPING_FILES}
        SECTIONS ${KAMPING_SECTIONS}
        LIBRARIES kamping_base
    )
    kamping_set_kassert_flags(${KAMPING_TARGET_NAME} ${ARGN})
endfunction ()
