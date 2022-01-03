list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
include(KaTestrophe)
include(KampingTestHelper)
include(GoogleTest)

# Registering tests without MPI:
kamping_register_test(test_checking_casts FILES test_checking_casts.cpp)

add_executable(test_buffer_factories test_buffer_factories.cpp)
target_link_libraries(test_buffer_factories PUBLIC gtest gtest_main gmock kamping)
gtest_discover_tests(test_buffer_factories WORKING_DIRECTORY ${PROJECT_DIR})
target_compile_options(test_buffer_factories PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(test_buffers test_buffers.cpp)
target_link_libraries(test_buffers PUBLIC gtest gtest_main gmock kamping)
gtest_discover_tests(test_buffers WORKING_DIRECTORY ${PROJECT_DIR})
target_compile_options(test_buffers PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(test_named_parameter_selection test_named_parameter_selection.cpp)
target_link_libraries(test_named_parameter_selection PUBLIC gtest gtest_main gmock kamping)
gtest_discover_tests(test_named_parameter_selection WORKING_DIRECTORY ${PROJECT_DIR})
target_compile_options(test_named_parameter_selection PRIVATE ${KAMPING_WARNING_FLAGS})


# Registering tests with MPI:

# We need MPI for the datatypes test as we are using MPI_Type_continuous which in
# turn requires MPI_Init() to be completed.
kamping_register_mpi_test(test_mpi_datatype FILES test_mpi_datatype.cpp CORES 1)

kamping_register_compilation_failure_test(
    test_mpi_datatype_unsupported_types
    FILES test_mpi_datatype_unsupported_types.cpp
    SECTIONS "POINTER" "FUNCTION" "UNION" "VOID"
)
