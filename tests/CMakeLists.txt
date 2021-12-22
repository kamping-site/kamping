list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
include(KaTestrophe)
include(KampingTestHelper)
include(GoogleTest)

# Registering tests without MPI:
kamping_register_test(test_checking_casts FILES test_checking_casts.cpp)

# Registering tests with MPI:

# We need MPI for the datatypes test as we are using MPI_Type_continuous which in
# turn requires MPI_Init() to be completed.
kamping_register_mpi_test(test_mpi_datatype FILES test_mpi_datatype.cpp CORES 1)

katestrophe_add_compilation_failure_test(
    TARGET test_mpi_datatype_unsupported_types
    FILES test_mpi_datatype_unsupported_types.cpp
    SECTIONS "POINTER" "FUNCTION" "UNION" "VOID"
    LIBRARIES kamping
)
