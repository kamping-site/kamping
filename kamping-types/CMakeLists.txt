cmake_minimum_required(VERSION 3.25)
project(kamping-types LANGUAGES CXX)

if (TARGET kamping::types)
    return()
endif ()

add_library(kamping_types INTERFACE)
target_include_directories(kamping_types INTERFACE include)
target_compile_features(kamping_types INTERFACE cxx_std_17)

find_package(MPI REQUIRED)
target_link_libraries(kamping_types INTERFACE MPI::MPI_CXX)

# kamping-types has its own kassert instance (macros KAMPING_TYPES_ASSERT*, header kamping/types/kassert/kassert.hpp),
# independent of KaMPIng's, as it is also used without KaMPIng (e.g. by kamping-v2). Its assertions check the results of
# MPI calls and are all at the normal level (30).
#
# The assertion level is KAMPING_TYPES_ASSERTION_LEVEL (a number, may be a generator expression). If empty, it is the
# level of KaMPIng when built as part of KaMPIng, and otherwise 30 in Debug mode and 10 (exceptions only) otherwise. It
# is the default level of the kamping_types::kassert instance, so it applies to every target linking to kamping::types;
# a single target can override it with the KAMPING_TYPES_ASSERT_ASSERTION_LEVEL property.
set(KAMPING_TYPES_ASSERTION_LEVEL
    ""
    CACHE STRING "Assertion level of kamping-types (empty: KaMPIng's level, or 30 in Debug mode and 10 otherwise)"
)
if (NOT KAMPING_TYPES_ASSERTION_LEVEL STREQUAL "")
    set(kamping_types_assertion_level ${KAMPING_TYPES_ASSERTION_LEVEL})
elseif (DEFINED KAMPING_TYPES_DEFAULT_ASSERTION_LEVEL)
    # set by KaMPIng before adding kamping-types
    set(kamping_types_assertion_level ${KAMPING_TYPES_DEFAULT_ASSERTION_LEVEL})
else ()
    set(kamping_types_assertion_level $<IF:$<CONFIG:Debug>,30,10>)
endif ()

if (NOT TARGET kamping_types::kassert)
    include(FetchContent)
    FetchContent_Declare(
        kamping_types_kassert
        GIT_REPOSITORY https://github.com/kamping-site/kassert
        GIT_TAG v1.1.0
    )
    block()
    set(KASSERT_VENDOR_ID kamping_types)
    set(KASSERT_CXX_NAMESPACE kamping::types::kassert)
    set(KASSERT_INCLUDE_SUBDIR kamping/types/kassert)
    set(KASSERT_PREFIX KAMPING_TYPES_ASSERT)
    set(KASSERT_DEFAULT_ASSERTION_LEVEL ${kamping_types_assertion_level})
    FetchContent_MakeAvailable(kamping_types_kassert)
    endblock()
endif ()
target_link_libraries(kamping_types INTERFACE kamping_types::kassert)

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
