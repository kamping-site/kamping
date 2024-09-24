add_executable(example_named_parameter_selection named_parameter_selection_example.cpp)
target_link_libraries(example_named_parameter_selection kamping)
target_compile_options(example_named_parameter_selection PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_reduce reduce_example.cpp)
target_link_libraries(example_reduce kamping)
target_compile_options(example_reduce PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_scatter scatter_example.cpp)
target_link_libraries(example_scatter kamping)
target_compile_options(example_scatter PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_alltoall alltoall_example.cpp)
target_link_libraries(example_alltoall kamping)
target_compile_options(example_alltoall PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_alltoallv alltoallv_example.cpp)
target_link_libraries(example_alltoallv kamping)
target_compile_options(example_alltoallv PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_gather gather_example.cpp)
target_link_libraries(example_gather kamping)
target_compile_options(example_gather PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_gatherv gatherv_example.cpp)
target_link_libraries(example_gatherv kamping)
target_compile_options(example_gatherv PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_allgather allgather_example.cpp)
target_link_libraries(example_allgather kamping)
target_compile_options(example_allgather PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_allgatherv allgatherv_example.cpp)
target_link_libraries(example_allgatherv kamping)
target_compile_options(example_allgatherv PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_bcast bcast_example.cpp)
target_link_libraries(example_bcast kamping)
target_compile_options(example_bcast PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_timer timer_example.cpp)
target_link_libraries(example_timer kamping)
target_compile_options(example_timer PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_request_pool request_pool_example.cpp)
target_link_libraries(example_request_pool kamping)
target_compile_options(example_timer PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_datatype datatype_example.cpp)
target_link_libraries(example_datatype kamping)
target_compile_options(example_datatype PRIVATE ${KAMPING_WARNING_FLAGS})

if (KAMPING_ENABLE_SERIALIZATION)
    add_executable(example_serialization serialization_example.cpp)
    target_link_libraries(example_serialization kamping cereal::cereal)
    target_compile_options(example_serialization PRIVATE ${KAMPING_WARNING_FLAGS})
endif ()

add_executable(example_paper paper_example.cpp)
target_link_libraries(example_paper kamping)
# these examples generate some warning due to conciseness, therefore we do not enable warnings
# target_compile_options(example_paper PRIVATE ${KAMPING_WARNING_FLAGS})
