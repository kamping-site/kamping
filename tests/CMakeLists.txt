list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
include(KampingTestHelper)
include(GoogleTest)

# Registering tests without MPI:
add_executable(demo_test demo_test.cpp)
target_link_libraries(demo_test PUBLIC gtest gtest_main gmock)
gtest_discover_tests(demo_test WORKING_DIRECTORY ${PROJECT_DIR})
target_compile_options(demo_test PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(test_checking_casts test_checking_casts.cpp)
target_link_libraries(test_checking_casts PUBLIC gtest gtest_main gmock kamping)
target_compile_options(test_checking_casts PRIVATE ${KAMPING_WARNING_FLAGS})
gtest_discover_tests(test_checking_casts WORKING_DIRECTORY ${PROJECT_DIR})

# Registering tests with MPI:
# to add a target for a test use kamping_add_test_executable
kamping_add_test_executable(demo_test_mpi FILES demo_test_mpi.cpp)

# if you want to test KaMPI.ng, you have to link it explicitly, like so:
# target_link_libraries(demo_test_mpi PRIVATE kamping)

# register the test using kamping_add_mpi_test
# this will register the test to run using 2, 4, 8, 16 MPI ranks
kamping_add_mpi_test(demo_test_mpi CORES 2 4 8 16)

# We need MPI for the datatypes test as we are using MPI_Type_continuous which in
# turn requires MPI_Init() to be completed.
kamping_add_test_executable(test_mpi_datatype FILES test_mpi_datatype.cpp)
target_link_libraries(test_mpi_datatype PRIVATE kamping)
kamping_add_mpi_test(test_mpi_datatype CORES 1)

kamping_add_compilation_failure_test(
    TARGET test_mpi_datatype
    FILES test_mpi_datatype_unsupported_types.cpp
    SECTIONS "POINTER" "FUNCTION" "UNION" "VOID"
    LIBRARIES kamping
)
