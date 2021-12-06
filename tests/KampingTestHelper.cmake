add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

# gtest-mpi-listener does not use modern CMake, therefore we need this fix
set(gtest-mpi-listener_SOURCE_DIR ${CMAKE_SOURCE_DIR}/extern/gtest-mpi-listener)
add_library(gtest-mpi-listener INTERFACE)
target_include_directories(gtest-mpi-listener INTERFACE "${gtest-mpi-listener_SOURCE_DIR}")
target_link_libraries(gtest-mpi-listener INTERFACE MPI::MPI_CXX gtest gmock)

# register the test main class
add_library(mpi-gtest-main EXCLUDE_FROM_ALL mpi-gtest-main.cpp)
target_link_libraries(mpi-gtest-main PUBLIC gtest-mpi-listener)

# keep the cache clean
mark_as_advanced(
  BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
  gmock_build_tests gtest_build_samples gtest_build_tests
  gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )

# Adds an executable target with the specified files and links gtest and the MPI gtest runner
function(kamping_add_test_executable KAMPING_TARGET)
    cmake_parse_arguments(
        "KAMPING"
        ""
        ""
        "FILES"
        ${ARGN}
        )
    add_executable(${KAMPING_TARGET} "${KAMPING_FILES}")
    target_link_libraries(${KAMPING_TARGET} PUBLIC gtest mpi-gtest-main)
    target_compile_options(${KAMPING_TARGET} PRIVATE ${KAMPING_WARNING_FLAGS})
endfunction()

# Registers an executable target as a test to be executed with the the specified number of MPI ranks
function(kamping_add_mpi_test KAMPING_TEST_TARGET)
    cmake_parse_arguments(
        KAMPING
        ""
        ""
        "CORES"
        ${ARGN}
        )
    if(NOT KAMPING_CORES)
        set(KAMPING_CORES ${MPIEXEC_MAX_NUMPROCS})
    endif()
    foreach(p ${KAMPING_CORES})
        if(${p} LESS_EQUAL ${MPIEXEC_MAX_NUMPROCS})
            set(TEST_NAME "${KAMPING_TEST_TARGET}.${p}cores")
            add_test(NAME "${TEST_NAME}"
              COMMAND
              ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${p} $<TARGET_FILE:${KAMPING_TEST_TARGET}>
              WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        endif()
    endforeach()
endfunction()
