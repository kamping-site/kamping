add_executable(example example.cpp)
target_link_libraries(example kamping)
target_compile_options(example PRIVATE ${KAMPING_WARNING_FLAGS})
