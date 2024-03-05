# Top-k reduce
add_executable(example_top_k_reduce top_k_reduce_example.cpp)
target_link_libraries(example_top_k_reduce kamping)
target_compile_options(example_top_k_reduce PRIVATE ${KAMPING_WARNING_FLAGS})

# Plugin: number of NUMA nodes
add_executable(example_my_num_numa_nodes my_num_numa_nodes_plugin.cpp)
target_link_libraries(example_my_num_numa_nodes kamping)
target_compile_options(example_my_num_numa_nodes PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_grid_alltoall grid_alltoall_plugin.cpp)
target_link_libraries(example_grid_alltoall kamping)
target_compile_options(example_grid_alltoall PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(sample_sort sample-sort/sample_sort.cpp)
target_link_libraries(sample_sort PUBLIC kamping)
target_compile_options(sample_sort PRIVATE ${KAMPING_WARNING_FLAGS})
