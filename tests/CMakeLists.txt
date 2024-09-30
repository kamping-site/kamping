list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(KaTestrophe)
include(KampingTestHelper)
include(GoogleTest)

set(KAMPING_TEST_TIMEOUT
    "20"
    CACHE STRING "Test timeout in seconds"
)
message(STATUS "Test timeout set to ${KAMPING_TEST_TIMEOUT} seconds.")

include(ProcessorCount)
ProcessorCount(N)
if (NOT N EQUAL 0)
    add_custom_target(
        check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --timeout ${KAMPING_TEST_TIMEOUT} -j ${N}
    )
endif ()

option(KAMPING_TEST_ENABLE_SANITIZERS "Enable undefined behavior sanitizer and address sanitizer." OFF)

# Registering tests without MPI:
kamping_register_test(test_checking_casts FILES checking_casts_test.cpp)
kamping_register_test(test_result FILES result_test.cpp)
kamping_register_test(test_nonblocking_result FILES nonblocking_result_test.cpp)
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
kamping_register_test(test_rank_ranges FILES rank_ranges_test.cpp)
kamping_register_test(test_measurement_utils FILES measurements/measurement_utils_test.cpp)
kamping_register_test(test_traits FILES utils/traits_test.cpp)
kamping_register_test(test_scoped_type FILES scoped_type_test.cpp)
kamping_register_test(test_communication_graph FILES communication_graph_test.cpp)

# Registering tests with MPI:

# We need MPI for the datatypes test as we are using MPI_Type_continuous which in turn requires MPI_Init() to be
# completed.
kamping_register_mpi_test(
    test_mpi_alltoall
    FILES collectives/mpi_alltoall_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_neighbor_alltoall
    FILES collectives/neighborhood/mpi_alltoall_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_alltoallv
    FILES collectives/mpi_alltoallv_test.cpp
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
    test_mpi_ibarrier
    FILES collectives/mpi_ibarrier_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_communicator
    FILES mpi_communicator_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_distributed_graph_communicator
    FILES distributed_graph_communicator_test.cpp
    CORES 4
)
kamping_register_mpi_test(
    test_mpi_group
    FILES mpi_group_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_datatype
    FILES mpi_datatype_test.cpp
    CORES 1 NO_GLIBCXX_DEBUG_CONTAINERS
)
kamping_register_mpi_test(
    test_mpi_gather
    FILES collectives/mpi_gather_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_gatherv
    FILES collectives/mpi_gatherv_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_send
    FILES p2p/mpi_send_test.cpp
    CORES 2 4
)
kamping_register_mpi_test(
    test_mpi_isend
    FILES p2p/mpi_isend_test.cpp
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
    test_mpi_iprobe
    FILES p2p/mpi_iprobe_test.cpp
    CORES 1 2 4
)
kamping_register_mpi_test(
    test_mpi_recv
    FILES p2p/mpi_recv_test.cpp
    CORES 1 2 4
)
kamping_register_mpi_test(
    test_mpi_try_recv
    FILES p2p/mpi_try_recv_test.cpp
    CORES 1 2 4
)
kamping_register_mpi_test(
    test_mpi_irecv
    FILES p2p/mpi_irecv_test.cpp
    CORES 1 2 4
)
kamping_register_mpi_test(
    test_mpi_allgather
    FILES collectives/mpi_allgather_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_allgatherv
    FILES collectives/mpi_allgatherv_test.cpp
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
    test_mpi_iallreduce
    FILES collectives/mpi_iallreduce_test.cpp
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
set(grid_test_cores 1 3 4 5 11 13)
kamping_register_mpi_test(
    test_alltoall_grid
    FILES plugins/alltoall_grid_test.cpp
    CORES ${grid_test_cores}
)
foreach (p ${grid_test_cores})
    set_tests_properties(test_alltoall_grid.${p}cores PROPERTIES TIMEOUT 40)
endforeach ()
kamping_register_mpi_test(
    test_alltoall_sparse
    FILES plugins/alltoall_sparse_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_sort
    FILES plugins/sort_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_alltoall_dispatch
    FILES plugins/alltoall_dispatch_test.cpp
    CORES 1 4
)
# kamping_register_mpi_test( test_reproducible_reduce FILES plugins/reproducible_reduce.cpp CORES 4 )
kamping_register_mpi_test(
    test_hooks
    FILES hooks_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_allocator
    FILES allocator_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_request_overriding_test_and_wait
    FILES request_test_overriding_test_and_wait.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_request
    FILES request_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_request_pool
    FILES request_pool_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_measurement_utils_with_communication
    FILES measurements/mpi_measurement_utils_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_timer
    FILES measurements/mpi_timer_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_counter
    FILES measurements/mpi_counter_test.cpp
    CORES 1 4
)
kamping_register_mpi_test(
    test_mpi_helpers_for_testing
    FILES helpers_for_testing_mpi_test.cpp
    CORES 1 4
)
if (KAMPING_ENABLE_SERIALIZATION)
    kamping_register_mpi_test(
        test_serialization
        FILES serialization_test.cpp
        CORES 1 2 4
    )
endif ()
kamping_register_mpi_test(
    test_flatten
    FILES utils/flatten_test.cpp
    CORES 1 4
)

kamping_register_mpi_test(
    test_examples_from_paper
    FILES examples_from_paper_test.cpp
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

# Test InitMPIMode::InitFinalizeIfNecessary: autoamtically decides if it needs to call MPI_Init() and MPI_Finalize()

# Not necessary, with argc and argv
add_executable(test_environment_init_finalize_if_necessary_0 environment_init_finalize_if_necessary_test.cpp)
target_link_libraries(test_environment_init_finalize_if_necessary_0 PRIVATE kamping_base MPI::MPI_CXX)
target_compile_definitions(test_environment_init_finalize_if_necessary_0 PRIVATE -DKAMPING_ENVIRONMENT_TEST_WITH_PARAM)
target_compile_definitions(test_environment_init_finalize_if_necessary_0 PRIVATE -D_GLIBCXX_DEBUG)
target_compile_definitions(test_environment_init_finalize_if_necessary_0 PRIVATE -D_GLIBCXX_DEBUG_PEDANTIC)
# Necessary, with argc and argv
add_executable(test_environment_init_finalize_if_necessary_1 environment_init_finalize_if_necessary_test.cpp)
target_link_libraries(test_environment_init_finalize_if_necessary_1 PRIVATE kamping_base MPI::MPI_CXX)
target_compile_definitions(
    test_environment_init_finalize_if_necessary_1 PRIVATE -DKAMPING_ENVIRONMENT_TEST_WITH_PARAM
                                                          -DKAMPING_ENVIRONMENT_TEST_INIT_FINALIZE_NECESSARY
)
target_compile_definitions(test_environment_init_finalize_if_necessary_1 PRIVATE -D_GLIBCXX_DEBUG)
target_compile_definitions(test_environment_init_finalize_if_necessary_1 PRIVATE -D_GLIBCXX_DEBUG_PEDANTIC)
# Not necessary, no argc and argv
add_executable(test_environment_init_finalize_if_necessary_2 environment_init_finalize_if_necessary_test.cpp)
target_link_libraries(test_environment_init_finalize_if_necessary_2 PRIVATE kamping_base MPI::MPI_CXX)
target_compile_definitions(test_environment_init_finalize_if_necessary_2 PRIVATE -DKAMPING_ENVIRONMENT_TEST_NO_PARAM)
target_compile_definitions(test_environment_init_finalize_if_necessary_2 PRIVATE -D_GLIBCXX_DEBUG)
target_compile_definitions(test_environment_init_finalize_if_necessary_2 PRIVATE -D_GLIBCXX_DEBUG_PEDANTIC)
# Necessary, no argc and argv
add_executable(test_environment_init_finalize_if_necessary_3 environment_init_finalize_if_necessary_test.cpp)
target_link_libraries(test_environment_init_finalize_if_necessary_3 PRIVATE kamping_base MPI::MPI_CXX)
target_compile_definitions(
    test_environment_init_finalize_if_necessary_3 PRIVATE -DKAMPING_ENVIRONMENT_TEST_NO_PARAM
                                                          -DKAMPING_ENVIRONMENT_TEST_INIT_FINALIZE_NECESSARY
)
target_compile_definitions(test_environment_init_finalize_if_necessary_3 PRIVATE -D_GLIBCXX_DEBUG)
target_compile_definitions(test_environment_init_finalize_if_necessary_3 PRIVATE -D_GLIBCXX_DEBUG_PEDANTIC)

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
             "ALLOC_NEW_NOT_DEDUCTABLE" "FAULTY_VECTOR_ALLOC_NEW_AUTO"
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
    test_mpi_alltoall_compilation_failure
    FILES collectives/mpi_alltoall_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT" "RECV_TYPE_GIVEN_BUT_NO_RECV_COUNT"
             "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT" "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_alltoallv_compilation_failure
    FILES collectives/mpi_alltoallv_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNTS" "RECV_TYPE_GIVEN_BUT_NO_RECV_COUNTS"
             "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT" "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_allgather_compilation_failure
    FILES collectives/mpi_allgather_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT" "RECV_TYPE_GIVEN_BUT_NO_RECV_COUNT"
             "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT" "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_allgatherv_compilation_failure
    FILES collectives/mpi_allgatherv_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT" "RECV_TYPE_GIVEN_BUT_NO_RECV_COUNT"
             "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT" "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_allreduce_compilation_failure
    FILES collectives/mpi_allreduce_compilation_failure_test.cpp
    SECTIONS "OPERATION_TYPE_DOES_NOT_MATCH_BUFFER_TYPE" "SEND_RECV_TYPE_GIVEN_BUT_NO_SEND_RECV_COUNT"
             "SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT"
             "SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY" "SINGLE_VARIANT_WITH_VECTOR"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_reduce_compilation_failure
    FILES collectives/mpi_reduce_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT" "SEND_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT"
             "SEND_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY" "SINGLE_VARIANT_WITH_VECTOR"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_bcast_compilation_failure
    FILES collectives/mpi_bcast_compilation_failure_test.cpp
    SECTIONS "SEND_RECV_COUNT_GIVEN" "SEND_RECV_TYPE_GIVEN_BUT_NO_SEND_RECV_COUNT"
             "SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT"
             "SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY" "SINGLE_VARIANT_WITH_VECTOR"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_gatherv_compilation_failure
    FILES collectives/mpi_gatherv_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT" "RECV_TYPE_GIVEN_BUT_NO_RECV_COUNTS"
             "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT" "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_exscan_compilation_failure
    FILES collectives/mpi_exscan_compilation_failure_test.cpp
    SECTIONS "SEND_RECV_TYPE_GIVEN_BUT_NO_SEND_RECV_COUNT" "SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT"
             "SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY" "SINGLE_VARIANT_WITH_VECTOR"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_scan_compilation_failure
    FILES collectives/mpi_scan_compilation_failure_test.cpp
    SECTIONS "SEND_RECV_TYPE_GIVEN_BUT_NO_SEND_RECV_COUNT" "SEND_RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT"
             "SEND_REC_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY" "SINGLE_VARIANT_WITH_VECTOR"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_scatterv_compilation_failure
    FILES collectives/mpi_scatterv_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNTS" "RECV_TYPE_GIVEN_BUT_NO_RECV_COUNT"
             "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT" "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_scatter_compilation_failure
    FILES collectives/mpi_scatter_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT" "RECV_TYPE_GIVEN_BUT_NO_RECV_COUNT"
             "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT" "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_gather_compilation_failure
    FILES collectives/mpi_gather_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT" "RECV_TYPE_GIVEN_BUT_NO_RECV_COUNT"
             "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT" "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_operation_builder_compilation_failure
    FILES operation_builder_compilation_failures_test.cpp
    SECTIONS "COPY_CONSTRUCT_OP_BUILDER_BUFFER" "COPY_ASSIGN_OP_BUILDER_BUFFER"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_recv_compilation_failure
    FILES p2p/mpi_recv_compilation_failure_test.cpp
    SECTIONS "OWNING_STATUS" "PROC_NULL" "RECV_COUNT_GIVEN" "RECV_BUF_GIVEN"
             "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_RESIZE_TO_FIT" "RECV_TYPE_GIVEN_BUT_RESIZE_POLICY_IS_GROW_ONLY"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_send_compilation_failure
    FILES p2p/mpi_send_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_STANDARD_MODE"
             "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_SYNCHRONOUS_MODE"
             "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_BUFFERED_MODE" "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_READY_MODE"
    LIBRARIES kamping_base
)
kamping_register_compilation_failure_test(
    test_mpi_isend_compilation_failure
    FILES p2p/mpi_isend_compilation_failure_test.cpp
    SECTIONS "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_STANDARD_MODE"
             "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_SYNCHRONOUS_MODE"
             "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_BUFFERED_MODE" "SEND_TYPE_GIVEN_BUT_NO_SEND_COUNT_IN_READY_MODE"
    LIBRARIES kamping_base
)
