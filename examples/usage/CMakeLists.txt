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

add_executable(example_sendrecv sendrecv_example.cpp)
target_link_libraries(example_sendrecv kamping)
target_compile_options(example_sendrecv PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_mdspan mdspan_example.cpp)
target_link_libraries(example_mdspan kamping)
target_compile_options(example_mdspan PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_adapter adapter_example.cpp)
target_link_libraries(example_adapter kamping)
target_compile_options(example_adapter PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_adapter_alltoallv alltoallv_adapter_example.cpp)
target_link_libraries(example_adapter_alltoallv kamping)
target_compile_options(example_adapter_alltoallv PRIVATE ${KAMPING_WARNING_FLAGS})

add_executable(example_recv recv_example.cpp)
target_link_libraries(example_recv kamping)
target_compile_options(example_recv PRIVATE ${KAMPING_WARNING_FLAGS})

if (KAMPING_ENABLE_SERIALIZATION)
    add_executable(example_serialization serialization_example.cpp)
    target_link_libraries(example_serialization kamping cereal::cereal)
    target_compile_options(example_serialization PRIVATE ${KAMPING_WARNING_FLAGS})
endif ()

add_executable(example_paper paper_example.cpp)
target_link_libraries(example_paper kamping)
# these examples generate some warning due to conciseness, therefore we do not enable warnings
# target_compile_options(example_paper PRIVATE ${KAMPING_WARNING_FLAGS})

# find_package(KokkosComm REQUIRED)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/releases/download/5.0.2/kokkos-5.0.2.tar.gz
  URL_HASH SHA256=188817bb452ca805ee8701f1c5adbbb4fb83dc8d1c50624566a18a719ba0fa5e
  OVERRIDE_FIND_PACKAGE
  SYSTEM
)
FetchContent_Declare(
  KokkosComm
  GIT_REPOSITORY https://github.com/kokkos/kokkos-comm.git
  SYSTEM
)
FetchContent_MakeAvailable(Kokkos KokkosComm)
add_library(KokkosComm::KokkosComm ALIAS KokkosComm)

add_executable(example_kokkos kokkos_example.cpp)
target_link_libraries(example_kokkos kamping KokkosComm::KokkosComm)
target_compile_options(example_kokkos PRIVATE ${KAMPING_WARNING_FLAGS})
