cmake_minimum_required(VERSION 3.1) #not sure what we really need
project(Kamping)

find_package(MPI REQUIRED)

add_library(kamping INTERFACE)
target_include_directories(kamping INTERFACE include)
# set_target_properties(kamping PROPERTIES
#   CXX_STANDARD 17
#   )
# TODO Make compiler independent
target_compile_options(kamping INTERFACE -std=c++17)
target_link_libraries(kamping INTERFACE MPI::MPI_CXX)

add_subdirectory(tests)
