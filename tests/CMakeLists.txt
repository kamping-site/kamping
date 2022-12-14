list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(KaTestrophe)
include(KampingTestHelper)
include(GoogleTest)

include(ProcessorCount)
ProcessorCount(N)
if (NOT N EQUAL 0)
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout 10 -j ${N})
endif ()

# Registering tests without MPI:
kamping_register_test(test_checking_casts FILES checking_casts_test.cpp)
kamping_register_test(test_mpi_function_wrapper_helpers FILES mpi_function_wrapper_helpers_test.cpp)
kamping_register_test(test_mpi_operations FILES mpi_operations_test.cpp)
kamping_register_test(test_named_parameter_check FILES named_parameter_check_test.cpp)
kamping_register_test(test_named_parameter_selection FILES named_parameter_selection_test.cpp)
kamping_register_test(test_named_parameters FILES named_parameters_test.cpp)
kamping_register_test(test_data_buffer FILES data_buffer_test.cpp)
kamping_register_test(test_parameter_objects FILES parameter_objects_test.cpp)
kamping_register_test(test_span FILES span_test.cpp)
kamping_register_test(test_helpers_for_testing FILES helpers_for_testing_test.cpp)
kamping_register_test(test_operation_builder FILES operation_builder_test.cpp)
kamping_register_test(test_has_member FILES has_member_test.cpp)

# Registering tests with MPI:

# We need MPI for the datatypes test as we are using MPI_Type_continuous which in turn requires MPI_Init() to be
# completed.
kamping_register_mpi_test(
    test_mpi_alltoall
    FILES collectives/mpi_alltoall_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_scatter
    FILES collectives/mpi_scatter_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_scatterv
    FILES collectives/mpi_scatterv_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_barrier
    FILES collectives/mpi_barrier_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_communicator
    FILES mpi_communicator_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_datatype
    FILES mpi_datatype_test.cpp
    CORES 1
)
kamping_register_mpi_test(
    test_mpi_gather
    FILES collectives/mpi_gather_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_send
    FILES p2p/mpi_send_test.cpp
    CORES 2 4
)
kamping_register_mpi_test(
    test_mpi_status
    FILES mpi_status_test.cpp
    CORES 2 4
)
kamping_register_mpi_test(
    test_mpi_probe
    FILES p2p/mpi_probe_test.cpp
    CORES 1 2 4
)
kamping_register_mpi_test(
    test_mpi_recv
    FILES p2p/mpi_recv_test.cpp
    CORES 1 2 4
)
kamping_register_mpi_test(
    test_mpi_allgather
    FILES collectives/mpi_allgather_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_reduce
    FILES collectives/mpi_reduce_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_bcast
    FILES collectives/mpi_bcast_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_allreduce
    FILES collectives/mpi_allreduce_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_scan
    FILES collectives/mpi_scan_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_exscan
    FILES collectives/mpi_exscan_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_operation_wrapper
    FILES mpi_operation_wrapper_test.cpp
    CORES 1
)
kamping_register_mpi_test(
    test_named_parameters_mpi
    FILES named_parameters_mpi_test.cpp
    CORES 1
)
kamping_register_mpi_test(
    test_multiple_compilation_units
    FILES multiple_compilation_units_test/main.cpp multiple_compilation_units_test/gatherer.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_environment
    FILES environment_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_communicator_is_same_on_all_ranks
    FILES comm_helper/mpi_communicator_is_same_on_all_ranks_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_implementation_helpers
    FILES implementation_helpers_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_plugins
    FILES plugins_test.cpp
    CORES 1 4
)

# Tests using C++ 20
kamping_register_mpi_test(
    test_std_span_alltoallv_cpp20
    FILES cpp20/std_span_alltoallv_test.cpp
    CORES 1 4
)
target_compile_features(test_std_span_alltoallv_cpp20 PRIVATE cxx_std_20)

# This should ensure that our Span implementation is consistent with C++ 20s std::span
kamping_register_test(test_span_cpp20 FILES span_test.cpp)
target_compile_features(test_span_cpp20 PRIVATE cxx_std_20)

# The environment test does not use our wrapper functions because it can't use our google test setup. First variant does
# only the minimum amount of interaction with kamping::Environment.
add_executable(test_environment_init_finalize_basic environment_init_finalize_test.cpp)
target_link_libraries(test_environment_init_finalize_basic PRIVATE kamping_base MPI::MPI_CXX)
target_compile_definitions(test_environment_init_finalize_basic PRIVATE -DKAMPING_ENVIRONMENT_TEST_NO_PARAM)
target_compile_definitions(test_environment_init_finalize_basic PRIVATE -D_GLIBCXX_DEBUG)
target_compile_definitions(test_environment_init_finalize_basic PRIVATE -D_GLIBCXX_DEBUG_PEDANTIC)
katestrophe_add_mpi_test(test_environment_init_finalize_basic CORES 1 4)
# Second variant passes parameters to MPI_Init and calls finallize manually.
add_executable(test_environment_init_finalize_manual environment_init_finalize_test.cpp)
target_link_libraries(test_environment_init_finalize_manual PRIVATE kamping_base MPI::MPI_CXX)
target_compile_definitions(
    test_environment_init_finalize_manual PRIVATE -DKAMPING_ENVIRONMENT_TEST_WITH_PARAM
                                                  -DKAMPING_ENVIRONMENT_TEST_EXPLICIT_FINALIZE
)
target_compile_definitions(test_environment_init_finalize_manual PRIVATE -D_GLIBCXX_DEBUG)
target_compile_definitions(test_environment_init_finalize_manual PRIVATE -D_GLIBCXX_DEBUG_PEDANTIC)
katestrophe_add_mpi_test(test_environment_init_finalize_manual CORES 1 4)

kamping_register_compilation_failure_test(
    test_mpi_datatype_unsupported_types
    FILES mpi_datatype_unsupported_types_test.cpp
    SECTIONS "POINTER" "FUNCTION" "UNION" "VOID"
)

kamping_register_compilation_failure_test(
    test_mpi_communicator_is_same_on_all_ranks_compilation_failure
    FILES comm_helper/mpi_communicator_is_same_on_all_ranks_compilation_failure_test.cpp
    SECTIONS "VALUE_IS_A_POINTER"
    LIBRARIES kamping_base
)

kamping_register_compilation_failure_test(
    test_named_parameter_selection_compilation_failure
    FILES named_parameter_selection_compilation_failures_test.cpp
    SECTIONS "REQUESTED_PARAMETER_NOT_GIVEN" "DEFAULT_ARGUMENTS_DO_NOT_MATCH"
    LIBRARIES kamping_base
)

kamping_register_compilation_failure_test(
    test_named_parameter_check_compilation_failure
    FILES named_parameter_check_compilation_failures_test.cpp
    SECTIONS "MISSING_REQUIRED_PARAMETER" "UNSUPPORTED_PARAMETER_NO_PARAMETERS"
             "UNSUPPORTED_PARAMETER_ONLY_OPTIONAL_PARAMETERS" "DUPLICATE_PARAMETERS"
    LIBRARIES kamping_base
)

kamping_register_compilation_failure_test(
    test_data_buffer_compilation_failure
    FILES data_buffer_compilation_failures_test.cpp
    SECTIONS "COPY_CONSTRUCT_CONTAINER_CONST_BUFFER"
             "COPY_ASSIGN_CONTAINER_CONST_BUFFER"
             "COPY_CONSTRUCT_SINGLE_ELMENT_CONST_BUFFER"
             "COPY_ASSIGN_SINGLE_ELMENT_CONST_BUFFER"
             "COPY_CONSTRUCT_SINGLE_ELMENT_MODIFIABLE_BUFFER"
             "COPY_ASSIGN_SINGLE_ELMENT_MODIFIABLE_BUFFER"
             "COPY_CONSTRUCT_USER_ALLOC_CONTAINER_BUFFER"
             "COPY_ASSIGN_USER_ALLOC_CONTAINER_BUFFER"
             "COPY_CONSTRUCT_LIB_ALLOC_CONTAINER_BUFFER"
             "COPY_ASSIGN_LIB_ALLOC_CONTAINER_BUFFER"
             "COPY_CONSTRUCT_LIB_ALLOC_SINGLE_ELEMENT_BUFFER"
             "COPY_ASSIGN_LIB_ALLOC_SINGLE_ELEMENT_BUFFER"
             "COPY_CONSTRUCT_ROOT_BUFFER"
             "COPY_ASSIGN_ROOT_BUFFER"
             "VALUE_CONSTRUCTOR_REFERENCING_DATA_BUFFER"
             "DEFAULT_CONSTRUCT_USER_ALLOCATED_DATA_BUFFER"
             "EXTRACT_USER_ALLOCATED_DATA_BUFFER"
             "RESIZE_CONST_DATA_BUFFER"
             "GET_SINGLE_ELEMENT_ON_VECTOR"
             "ACCESS_CONST_VECTOR_BOOL"
             "ACCESS_VECTOR_BOOL"
    LIBRARIES kamping_base
)

kamping_register_compilation_failure_test(
    test_named_parameters_compilation_failure
    FILES named_parameters_compilation_failure_test.cpp
    SECTIONS "VECTOR_BOOL_LVALUE" "VECTOR_BOOL_RVALUE" "VECTOR_BOOL_CUSTOM_ALLOCATOR" "VECTOR_BOOL_NEW_CONTAINER"
    LIBRARIES kamping_base
)

kamping_register_compilation_failure_test(
    test_vector_bool_compilation_failure
    FILES vector_bool_compilation_failures_test.cpp
    SECTIONS "SINGLE_BOOL_VEC_BOOL" "SINGLE_KABOOL_VEC_BOOL" "SEND_VEC_BOOL" "SEND_VEC_KABOOL_RECV_VEC_BOOL"
    LIBRARIES kamping_base
)

kamping_register_compilation_failure_test(
    test_required_type_compilation_failure
    FILES required_type_compilation_failure_test.cpp
    SECTIONS "RECV_COUNT_OUT_PASSED"
             "RECV_COUNT_OUT_NEW_CONTAINER"
             "RECV_COUNTS_PASSED"
             "RECV_COUNTS_OUT_PASSED"
             "RECV_COUNTS_OUT_NEW_CONTAINER"
             "SEND_COUNTS_PASSED"
             "RECV_DISPLS_PASSED"
             "RECV_DISPLS_OUT_PASSED"
             "RECV_DISPLS_OUT_NEW_CONTAINER"
             "SEND_DISPLS_PASSED"
             "SEND_DISPLS_OUT_PASSED"
             "SEND_DISPLS_OUT_NEW_CONTAINER"
    LIBRARIES kamping_base
)

kamping_register_compilation_failure_test(
    test_mpi_bcast_compilation_failure
    FILES collectives/mpi_bcast_compilation_failure_test.cpp
    SECTIONS "RECV_COUNT_GIVEN"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_operation_builder_compilation_failure
    FILES operation_builder_compilation_failures_test.cpp
    SECTIONS "COPY_CONSTRUCT_OP_BUILDER_BUFFER" "COPY_ASSIGN_OP_BUILDER_BUFFER"
    LIBRARIES kamping_base
)
