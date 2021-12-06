add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

# gtest-mpi-listener does not use modern CMake, therefore we need this fix
set(gtest-mpi-listener_SOURCE_DIR ${CMAKE_SOURCE_DIR}/extern/gtest-mpi-listener)
add_library(gtest-mpi-listener INTERFACE)
target_include_directories(gtest-mpi-listener INTERFACE "${gtest-mpi-listener_SOURCE_DIR}")
target_link_libraries(gtest-mpi-listener INTERFACE MPI::MPI_CXX gtest gmock)


function(kamping_has_oversubscribe KAMPING_OVERSUBSCRIBE_FLAG)
  string(FIND ${MPI_CXX_LIBRARY_VERSION_STRING} "OpenMPI" SEARCH_POSITION1)
  string(FIND ${MPI_CXX_LIBRARY_VERSION_STRING} "Open MPI" SEARCH_POSITION2)
  # only Open MPI seems to require the --oversubscribe flag
  if(${SEARCH_POSITION1} EQUAL -1 AND ${SEARCH_POSITION2} EQUAL -1)
    set("${KAMPING_OVERSUBSCRIBE_FLAG}" "" PARENT_SCOPE)
  else()
    set("${KAMPING_OVERSUBSCRIBE_FLAG}" "--oversubscribe" PARENT_SCOPE)
  endif()
endfunction()
kamping_has_oversubscribe(MPIEXEC_OVERSUBSCRIBE_FLAG)

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
    set(TEST_NAME "${KAMPING_TEST_TARGET}.${p}cores")
    add_test(
      NAME "${TEST_NAME}"
      COMMAND
      ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${p} ${MPIEXEC_OVERSUBSCRIBE_FLAG} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${KAMPING_TEST_TARGET}> ${MPIEXEC_POSTFLAGS}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
  endforeach()
endfunction()
