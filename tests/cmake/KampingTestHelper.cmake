include(FetchContent)
FetchContent_Declare(
    KaTestrophe
    GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
    GIT_TAG v1.0.2
)
FetchContent_MakeAvailable(KaTestrophe)

# Registers a set of tests which should fail to compile.
#
# TARGET prefix for the targets to be built FILES the list of files to include in the target SECTIONS sections of the
# compilation test to build LIBRARIES libraries to link via target_link_libraries(...)
#
# Loosely based on: https://stackoverflow.com/questions/30155619/expected-build-failure-tests-in-cmake
function (katestrophe_add_compilation_failure_test)
    cmake_parse_arguments(
        "KATESTROPHE" # prefix
        "" # options
        "TARGET" # one value arguments
        "FILES;SECTIONS;LIBRARIES" # multiple value arguments
        ${ARGN}
    )

    # the file should compile without any section enabled
    add_executable(${KATESTROPHE_TARGET} ${KATESTROPHE_FILES})
    target_link_libraries(${KATESTROPHE_TARGET} PUBLIC GTest::gtest ${KATESTROPHE_LIBRARIES})

    # For each given section, add a target.
    foreach (SECTION ${KATESTROPHE_SECTIONS})
        string(TOLOWER ${SECTION} SECTION_LOWERCASE)
        set(THIS_TARGETS_NAME "${KATESTROPHE_TARGET}.${SECTION_LOWERCASE}")

        # Add the executable and link the libraries.
        add_executable(${THIS_TARGETS_NAME} ${KATESTROPHE_FILES})
        target_link_libraries(${THIS_TARGETS_NAME} PUBLIC gtest ${KATESTROPHE_LIBRARIES})

        # Select the correct section of the target by setting the appropriate preprocessor define.
        target_compile_definitions(${THIS_TARGETS_NAME} PRIVATE ${SECTION})

        # Exclude the target fromn the "all" target.
        set_target_properties(${THIS_TARGETS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)

        # Add a test invoking "cmake --build" to test if the target compiles.
        add_test(
            NAME "${THIS_TARGETS_NAME}"
            COMMAND cmake --build . --target ${THIS_TARGETS_NAME} --config $<CONFIGURATION>
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )

        # Specify, that the target should not compile.
        set_tests_properties("${THIS_TARGETS_NAME}" PROPERTIES WILL_FAIL TRUE)
    endforeach ()
endfunction ()

add_library(kamping_test_base INTERFACE)
if (KAMPING_CXX_FLAGS)
    target_compile_options(kamping_test_base INTERFACE ${KAMPING_CXX_FLAGS})
endif ()

function (kamping_set_kassert_flags KAMPING_TARGET_NAME)
    cmake_parse_arguments("KAMPING" "NO_EXCEPTION_MODE" "" "" ${ARGN})

    # Use global assertion level
    target_compile_definitions(
        ${KAMPING_TARGET_NAME} PRIVATE -DKAMPING_ASSERT_ASSERTION_LEVEL=${KAMPING_ASSERT_ASSERTION_LEVEL}
    )

    # Explicitly specify exception mode for tests, default to no exception mode
    if (NOT KAMPING_NO_EXCEPTION_MODE)
        target_compile_definitions(${KAMPING_TARGET_NAME} PRIVATE -DKAMPING_ASSERT_EXCEPTION_MODE)
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
    target_link_libraries(${KAMPING_TARGET_NAME} PRIVATE GTest::gtest_main GTest::gmock kamping_base kamping_test_base)
    gtest_discover_tests(${KAMPING_TARGET_NAME} WORKING_DIRECTORY ${PROJECT_BINARY_DIR} DISCOVERY_MODE PRE_TEST)
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
    target_link_libraries(${KAMPING_TARGET_NAME} PRIVATE kamping_base kamping_test_base)
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
