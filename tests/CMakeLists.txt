list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(KaTestrophe)
include(KampingTestHelper)
include(GoogleTest)

# Registering tests without MPI:
kamping_register_test(test_checking_casts FILES checking_casts_test.cpp)
kamping_register_test(test_kassert_exception_mode FILES kassert_test.cpp)
kamping_register_test(test_kassert_assertion_mode NO_EXCEPTION_MODE FILES kassert_test.cpp)
kamping_register_test(test_parameter_factories FILES parameter_factories_test.cpp)
kamping_register_test(test_parameter_objects FILES parameter_objects_test.cpp)
kamping_register_test(test_named_parameter_selection FILES named_parameter_selection_test.cpp)
kamping_register_test(test_mpi_function_wrapper_helpers FILES mpi_function_wrapper_helpers_test.cpp)
kamping_register_test(test_mpi_operations FILES mpi_operations_test.cpp)

# Registering tests with MPI:

# We need MPI for the datatypes test as we are using MPI_Type_continuous which in
# turn requires MPI_Init() to be completed.
kamping_register_mpi_test(test_operation_wrapper FILES mpi_operation_wrapper_test.cpp CORES 1)
kamping_register_mpi_test(test_parameter_factories_mpi FILES parameter_factories_mpi_test.cpp CORES 1)
kamping_register_mpi_test(test_mpi_datatype FILES mpi_datatype_test.cpp CORES 1)
kamping_register_mpi_test(test_mpi_communicator FILES mpi_communicator_test.cpp CORES 1 2 4)
kamping_register_mpi_test(test_mpi_alltoall FILES collectives/mpi_alltoall_test.cpp CORES 1 2 4)
kamping_register_mpi_test(test_mpi_reduce FILES collectives/mpi_reduce_test.cpp CORES 1 2 4)
kamping_register_mpi_test(test_mpi_barrier FILES collectives/mpi_barrier_test.cpp CORES 1 2 4)


kamping_register_compilation_failure_test(
    test_mpi_datatype_unsupported_types
    FILES mpi_datatype_unsupported_types_test.cpp
    SECTIONS "POINTER" "FUNCTION" "UNION" "VOID"
)

kamping_register_compilation_failure_test(
  test_mpi_function_wrapper_compilation_failure
  FILES mpi_function_wrapper_helpers_compilation_failures_test.cpp
  SECTIONS "RECV_BUFFER_NOT_EXTRACTABLE" "RECV_COUNTS_NOT_EXTRACTABLE" "RECV_DISPLACEMENTS_NOT_EXTRACTABLE" "SEND_DISPLACEMENTS_NOT_EXTRACTABLE"
  LIBRARIES kamping_base
)

kamping_register_compilation_failure_test(
  test_named_parameter_selection_compilation_failure
  FILES named_parameter_selection_compilation_failures_test.cpp
  SECTIONS "REQUESTED_PARAMETER_NOT_GIVEN" "DEFAULT_ARGUMENTS_DO_NOT_MATCH"
  LIBRARIES kamping_base
)
