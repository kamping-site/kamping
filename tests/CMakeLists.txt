list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
include(KaTestrophe)
include(KampingTestHelper)
include(GoogleTest)

# Registering tests without MPI:
kamping_register_test(test_checking_casts FILES test_checking_casts.cpp)

add_executable(test_parameter_factories test_parameter_factories.cpp)
target_link_libraries(test_parameter_factories PUBLIC gtest gtest_main gmock kamping)
gtest_discover_tests(test_parameter_factories WORKING_DIRECTORY ${PROJECT_DIR})
target_compile_options(test_parameter_factories PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(test_parameter_objects test_parameter_objects.cpp)
target_link_libraries(test_parameter_objects PUBLIC gtest gtest_main gmock kamping)
gtest_discover_tests(test_parameter_objects WORKING_DIRECTORY ${PROJECT_DIR})
target_compile_options(test_parameter_objects PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(test_named_parameter_selection test_named_parameter_selection.cpp)
target_link_libraries(test_named_parameter_selection PUBLIC gtest gtest_main gmock kamping)
gtest_discover_tests(test_named_parameter_selection WORKING_DIRECTORY ${PROJECT_DIR})
target_compile_options(test_named_parameter_selection PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(test_mpi_function_wrapper_helpers test_mpi_function_wrapper_helpers.cpp)
target_link_libraries(test_mpi_function_wrapper_helpers PUBLIC gtest gtest_main gmock kamping)
gtest_discover_tests(test_mpi_function_wrapper_helpers WORKING_DIRECTORY ${PROJECT_DIR})
target_compile_options(test_mpi_function_wrapper_helpers PRIVATE ${KAMPING_WARNING_FLAGS})

# Registering tests with MPI:

# We need MPI for the datatypes test as we are using MPI_Type_continuous which in
# turn requires MPI_Init() to be completed.
kamping_register_mpi_test(test_mpi_datatype FILES test_mpi_datatype.cpp CORES 1)

kamping_register_compilation_failure_test(
    test_mpi_datatype_unsupported_types
    FILES test_mpi_datatype_unsupported_types.cpp
    SECTIONS "POINTER" "FUNCTION" "UNION" "VOID"
)

kamping_add_compilation_failure_test(
  TARGET test_mpi_function_wrapper_compilation_failure
  FILES test_mpi_function_wrapper_helpers_compilation_failures.cpp
  SECTIONS "RECV_BUFFER_NOT_EXTRACTABLE" "RECV_COUNTS_NOT_EXTRACTABLE" "RECV_DISPLACEMENTS_NOT_EXTRACTABLE" "SEND_DISPLACEMENTS_NOT_EXTRACTABLE"
  LIBRARIES kamping
  )
