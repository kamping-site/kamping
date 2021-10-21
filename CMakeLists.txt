cmake_minimum_required(VERSION 3.1) #not sure what we really need
project(Kamping)

find_package(MPI REQUIRED)

add_library(kamping INTERFACE)
target_include_directories(kamping INTERFACE include SYSTEM ${MPI_INCLUDE_PATH}) 
target_link_libraries(kamping INTERFACE ${MPI_LIBRARIES})

add_subdirectory(tests)
