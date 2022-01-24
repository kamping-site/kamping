list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
include(KaTestrophe)
include(KampingTestHelper)
include(GoogleTest)

# Registering tests without MPI:
kamping_register_test(test_checking_casts FILES test_checking_casts.cpp)
kamping_register_test(test_mpi_operations FILES test_mpi_operations.cpp)
kamping_register_test(test_kassert FILES test_kassert.cpp)
kamping_register_test(test_parameter_factories FILES test_parameter_factories.cpp)
kamping_register_test(test_parameter_objects FILES test_parameter_objects.cpp)
kamping_register_test(test_named_parameter_selection FILES test_named_parameter_selection.cpp)
kamping_register_test(test_mpi_function_wrapper_helpers FILES test_mpi_function_wrapper_helpers.cpp)

# Registering tests with MPI:

# We need MPI for the datatypes test as we are using MPI_Type_continuous which in
# turn requires MPI_Init() to be completed.
kamping_register_mpi_test(test_mpi_datatype FILES test_mpi_datatype.cpp CORES 1)
kamping_register_mpi_test(test_mpi_communicator FILES test_mpi_communicator.cpp CORES 1 2 4)


kamping_register_compilation_failure_test(
    test_mpi_datatype_unsupported_types
    FILES test_mpi_datatype_unsupported_types.cpp
    SECTIONS "POINTER" "FUNCTION" "UNION" "VOID"
)

kamping_register_compilation_failure_test(
  test_mpi_function_wrapper_compilation_failure
  FILES test_mpi_function_wrapper_helpers_compilation_failures.cpp
  SECTIONS "RECV_BUFFER_NOT_EXTRACTABLE" "RECV_COUNTS_NOT_EXTRACTABLE" "RECV_DISPLACEMENTS_NOT_EXTRACTABLE" "SEND_DISPLACEMENTS_NOT_EXTRACTABLE"
  LIBRARIES kamping
)

kamping_register_compilation_failure_test(
  test_named_parameter_selection_compilation_failure
  FILES test_named_parameter_selection_compilation_failures.cpp
  SECTIONS "REQUESTED_PARAMETER_NOT_GIVEN"
  LIBRARIES kamping
)
