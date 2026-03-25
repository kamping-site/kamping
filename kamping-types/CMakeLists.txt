cmake_minimum_required(VERSION 3.25)
project(kamping-types LANGUAGES CXX)

add_library(kamping_types INTERFACE)
target_include_directories(kamping_types INTERFACE include)
target_compile_features(kamping_types INTERFACE cxx_std_17)

find_package(MPI REQUIRED)
target_link_libraries(kamping_types INTERFACE MPI::MPI_CXX)

# kassert (guard against double-fetch when used inside KaMPIng)
if (NOT TARGET kamping::kassert)
    include(FetchContent)
    FetchContent_Declare(
        kamping_kassert
        GIT_REPOSITORY https://github.com/kamping-site/kassert
        GIT_TAG v1.0.0
    )
    set(KASSERT_VENDOR_ID kamping)
    set(KASSERT_CXX_NAMESPACE kamping::kassert)
    set(KASSERT_INCLUDE_SUBDIR kamping/kassert)
    set(KASSERT_PREFIX KAMPING_ASSERT)
    FetchContent_MakeAvailable(kamping_kassert)
endif ()
target_link_libraries(kamping_types INTERFACE kamping::kassert)

# Optional reflection support
option(KAMPING_TYPES_ENABLE_REFLECTION "Enable struct reflection via Boost.PFR" OFF)
if (KAMPING_TYPES_ENABLE_REFLECTION)
    if (NOT TARGET Boost::pfr)
        include(FetchContent)
        FetchContent_Declare(
            pfr
            GIT_REPOSITORY https://github.com/boostorg/pfr
            GIT_TAG 2.2.0
            SYSTEM FIND_PACKAGE_ARGS 2.2.0
        )
        FetchContent_MakeAvailable(pfr)
    endif ()
    target_link_libraries(kamping_types INTERFACE Boost::pfr)
    target_compile_definitions(kamping_types INTERFACE KAMPING_ENABLE_REFLECTION)
endif ()

add_library(kamping::types ALIAS kamping_types)

option(KAMPING_TYPES_BUILD_EXAMPLES "Build kamping-types examples" ${PROJECT_IS_TOP_LEVEL})
if (KAMPING_TYPES_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
