# Top-k reduce
add_executable(example_top_k_reduce top_k_reduce_example.cpp)
target_link_libraries(example_top_k_reduce kamping)
target_compile_options(example_top_k_reduce PRIVATE ${KAMPING_WARNING_FLAGS})

# Plugin: number of NUMA nodes
add_executable(example_my_num_numa_nodes my_num_numa_nodes_plugin.cpp)
target_link_libraries(example_my_num_numa_nodes kamping)
target_compile_options(example_my_num_numa_nodes PRIVATE ${KAMPING_WARNING_FLAGS})
