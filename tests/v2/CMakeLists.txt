kamping_register_test(test_v2_auto_counts_view FILES auto_counts_view_test.cpp)

kamping_register_test(test_v2_sentinels FILES sentinels_test.cpp)

kamping_register_mpi_test(
    test_v2_type_pool
    FILES v2_type_pool_test.cpp
    CORES 1 2
)

kamping_register_mpi_test(
    test_v2_allgatherv
    FILES v2_allgatherv_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_alltoall
    FILES v2_alltoall_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_alltoallv
    FILES v2_alltoallv_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_allreduce
    FILES v2_allreduce_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_reduce
    FILES v2_reduce_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_gather
    FILES v2_gather_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_gatherv
    FILES v2_gatherv_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_scatter
    FILES v2_scatter_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_scatterv
    FILES v2_scatterv_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_request_view
    FILES request_view_test.cpp
    CORES 1 2
)

kamping_register_mpi_test(
    test_v2_scan
    FILES v2_scan_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_exscan
    FILES v2_exscan_test.cpp
    CORES 1 2 4
)

kamping_register_mpi_test(
    test_v2_info
    FILES info_test.cpp
    CORES 1
)
