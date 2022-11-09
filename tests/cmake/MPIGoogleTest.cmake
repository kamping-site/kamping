# Adapted by the KaMPIng authors from GoogleTest.cmake included in CMake.
#
# Original license information: Distributed under the OSI-approved BSD 3-Clause License. See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

function (katestrophe_discover_tests TARGET)
    cmake_parse_arguments(
        "" "NO_PRETTY_TYPES;NO_PRETTY_VALUES"
        "TEST_PREFIX;TEST_SUFFIX;WORKING_DIRECTORY;TEST_LIST;DISCOVERY_TIMEOUT;XML_OUTPUT_DIR;MPI_EXEC_COMMAND"
        "EXTRA_ARGS;PROPERTIES;TEST_FILTER" ${ARGN}
    )

    if (NOT _WORKING_DIRECTORY)
        set(_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endif ()
    if (NOT _TEST_LIST)
        set(_TEST_LIST "${TARGET}_TESTS")
    endif ()
    if (NOT _DISCOVERY_TIMEOUT)
        set(_DISCOVERY_TIMEOUT 5)
    endif ()
    if (NOT _MPI_EXEC_COMMAND)
        set(_MPI_EXEC_COMMAND "mpiexec")
    endif ()

    get_property(
        has_counter
        TARGET ${TARGET}
        PROPERTY CTEST_DISCOVERED_TEST_COUNTER
        SET
    )
    if (has_counter)
        get_property(
            counter
            TARGET ${TARGET}
            PROPERTY CTEST_DISCOVERED_TEST_COUNTER
        )
        math(EXPR counter "${counter} + 1")
    else ()
        set(counter 1)
    endif ()
    set_property(TARGET ${TARGET} PROPERTY CTEST_DISCOVERED_TEST_COUNTER ${counter})

    # Define rule to generate test list for aforementioned test executable
    set(ctest_file_base "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}[${counter}]")
    set(ctest_include_file "${ctest_file_base}_include.cmake")
    set(ctest_tests_file "${ctest_file_base}_tests.cmake")
    message(STATUS ${ctest_tests_file})
    get_property(
        crosscompiling_emulator
        TARGET ${TARGET}
        PROPERTY CROSSCOMPILING_EMULATOR
    )

    add_custom_command(
        TARGET ${TARGET}
        POST_BUILD
        BYPRODUCTS "${ctest_tests_file}"
        COMMAND
            "${CMAKE_COMMAND}" "-D TEST_TARGET=${TARGET}" "-D TEST_EXECUTABLE=$<TARGET_FILE:${TARGET}>"
            "-D TEST_MPI_EXEC_COMMAND=${_MPI_EXEC_COMMAND}" "-D TEST_EXECUTOR=${crosscompiling_emulator}"
            "-D TEST_WORKING_DIR=${_WORKING_DIRECTORY}" "-D TEST_EXTRA_ARGS=${_EXTRA_ARGS}"
            "-D TEST_PROPERTIES=${_PROPERTIES}" "-D TEST_PREFIX=${_TEST_PREFIX}" "-D TEST_SUFFIX=${_TEST_SUFFIX}"
            "-D TEST_FILTER=${_TEST_FILTER}" "-D NO_PRETTY_TYPES=${_NO_PRETTY_TYPES}"
            "-D NO_PRETTY_VALUES=${_NO_PRETTY_VALUES}" "-D TEST_LIST=${_TEST_LIST}" "-D CTEST_FILE=${ctest_tests_file}"
            "-D TEST_DISCOVERY_TIMEOUT=${_DISCOVERY_TIMEOUT}" "-D TEST_XML_OUTPUT_DIR=${_XML_OUTPUT_DIR}"
            "-P ${_KATESTROPHE_DISCOVER_TESTS_SCRIPT}"
        VERBATIM
    )

    file(
        WRITE "${ctest_include_file}"
        "if(EXISTS \"${ctest_tests_file}\")\n" #
        "  include(\"${ctest_tests_file}\")\n" #
        "else()\n"
        "  add_test(${TARGET}_NOT_BUILT ${TARGET}_NOT_BUILT)\n" #
        "endif()\n" #
    )

    # Add discovered tests to directory TEST_INCLUDE_FILES
    set_property(
        DIRECTORY
        APPEND
        PROPERTY TEST_INCLUDE_FILES "${ctest_include_file}"
    )

endfunction ()

# -------------------------------------------

set(_KATESTROPHE_DISCOVER_TESTS_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/MPIGoogleTestAddTests.cmake")
