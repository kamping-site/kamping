cmake_minimum_required(VERSION 3.25)

project(
    KaMPIng
    DESCRIPTION "Flexible and (near) zero-overhead C++ bindings for MPI"
    LANGUAGES CXX
    VERSION 0.1.2
)

option(KAMPING_BUILD_TESTS "Build the tests for KaMPIng." ${PROJECT_IS_TOP_LEVEL})
option(KAMPING_BUILD_EXAMPLES "Build the example for KaMPIng." ${PROJECT_IS_TOP_LEVEL})
option(KAMPING_BUILD_DOCS "Build the docs for KaMPIng." OFF)
option(KAMPING_TESTS_DISCOVER OFF)
option(KAMPING_ENABLE_ULFM "Enable User-Level Failure-Mitigation (ULFM)" OFF)
option(KAMPING_ENABLE_SERIALIZATION "Enable support for serialization (requires Cereal)" ON)
option(KAMPING_ENABLE_REFLECTION "Enable support for reflecting struct members (requires Boost.PFR)" ON)
option(
    KAMPING_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR
    "Use Boost.PFR from system installed Boost, instead of using a standalone PFR install or building PFR from source."
    OFF
)
set(KAMPING_CXX_FLAGS
    ""
    CACHE STRING "Additional compilation flags to use for building tests and examples."
)

if (KAMPING_BUILD_DOCS)
    add_subdirectory(docs)
endif ()

set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
find_package(MPI REQUIRED)

add_library(kamping_base INTERFACE)
target_include_directories(kamping_base INTERFACE include)

# set C++ standard to C++17
target_compile_features(kamping_base INTERFACE cxx_std_17)
target_link_libraries(kamping_base INTERFACE MPI::MPI_CXX)

# Target for user-code
add_library(kamping INTERFACE)
target_link_libraries(kamping INTERFACE kamping_base)

# If enabled, use exceptions, otherwise use KAMPING_ASSERT()
option(KAMPING_EXCEPTION_MODE "Use exceptions to report recoverable errors." ON)
if (KAMPING_EXCEPTION_MODE)
    target_compile_definitions(kamping INTERFACE -DKAMPING_ASSERT_EXCEPTION_MODE)
    message(STATUS "Build with exceptions enabled.")
else ()
    message(STATUS "Build with exceptions disabled. Assertions are used instead.")
endif ()

# The assertion level controls which assertions are enabled during runtime:
#
# * Level 0: Disable all assertions
# * Level 10: Exception assertions = only enable exceptions (if not in exception mode)
# * Level 20: Light assertions = assertions that do not affect the running time of library operations significantly.
# * Level 30: Normal assertions = assertions that might slow down some operations of the library by a constant factor.
#   Should only be used in debug mode.
# * Level 40: Light communication assertions = assertions that perform additional communication causing small running
#   time overheads.
# * Level 50: Heavy communication assertions = assertions that perform additional communication causing significant
#   running time overheads.
# * Level 60: Heavy assertions = assertions that introduce overhead which renders some library operations infeasible
#   when invoked with any significant work load.
#
# Assertion levels can be set explicitly using the -DKAMPING_ASSERTION_LEVEL=... flag. If no level is set explicitly, we
# set it to 10 (exceptions only) in Release mode and 30 (up to normal assertions) in Debug mode.
set(KAMPING_ASSERTION_LEVEL
    $<IF:$<CONFIG:Debug>,"normal","exceptions">
    CACHE STRING "Assertion level"
)
set_property(
    CACHE KAMPING_ASSERTION_LEVEL
    PROPERTY STRINGS
             none
             exceptions
             light
             normal
             light_communication
             heavy_communication
             heavy
)
message(STATUS "Assertion level: ${KAMPING_ASSERTION_LEVEL}")

# If KAMPING_ASSERTION_LEVEL defaults to the generator expression, ${KAMPING_ASSERTION_LEVEL} may not be quoted However,
# if it is explicitly set to some constant string, it must be quoted Thus, all levels are listed twice, once with and
# without quotes @todo find a better solution for this problem
string(
    CONCAT KAMPING_ASSERT_ASSERTION_LEVEL
           $<$<STREQUAL:${KAMPING_ASSERTION_LEVEL},"none">:0>
           $<$<STREQUAL:"${KAMPING_ASSERTION_LEVEL}","none">:0>
           $<$<STREQUAL:${KAMPING_ASSERTION_LEVEL},"exceptions">:10>
           $<$<STREQUAL:"${KAMPING_ASSERTION_LEVEL}","exceptions">:10>
           $<$<STREQUAL:${KAMPING_ASSERTION_LEVEL},"light">:20>
           $<$<STREQUAL:"${KAMPING_ASSERTION_LEVEL}","light">:20>
           $<$<STREQUAL:${KAMPING_ASSERTION_LEVEL},"normal">:30>
           $<$<STREQUAL:"${KAMPING_ASSERTION_LEVEL}","normal">:30>
           $<$<STREQUAL:${KAMPING_ASSERTION_LEVEL},"light_communication">:40>
           $<$<STREQUAL:"${KAMPING_ASSERTION_LEVEL}","light_communication">:40>
           $<$<STREQUAL:${KAMPING_ASSERTION_LEVEL},"heavy_communication">:50>
           $<$<STREQUAL:"${KAMPING_ASSERTION_LEVEL}","heavy_communication">:50>
           $<$<STREQUAL:${KAMPING_ASSERTION_LEVEL},"heavy">:60>
           $<$<STREQUAL:"${KAMPING_ASSERTION_LEVEL}","heavy">:60>
)
target_compile_definitions(kamping INTERFACE -DKAMPING_ASSERT_ASSERTION_LEVEL=${KAMPING_ASSERT_ASSERTION_LEVEL})

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

target_link_libraries(kamping_base INTERFACE kamping::kassert)

FetchContent_Declare(
    pfr
    GIT_REPOSITORY https://github.com/boostorg/pfr
    GIT_TAG 2.2.0
    SYSTEM FIND_PACKAGE_ARGS 2.2.0
)

if (KAMPING_ENABLE_REFLECTION)
    if (KAMPING_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR)
        find_package(Boost 1.75 COMPONENTS headers CONFIG)
        if (NOT Boost_FOUND)
            message(
                FATAL_ERROR
                    "Boost.PFR: No compatible Boost version found. Use KAMPING_REFLECTION_USE_SYSTEM_BOOST_FOR_PFR=OFF to use standalone Boost.PFR."
            )
        else ()
            message(STATUS "Found Boost ${Boost_VERSION}: ${Boost_DIR}")
            message(STATUS "Using system Boost for Boost.PFR")
            add_library(kamping_pfr INTERFACE)
            # when using system installed Boost, it does not provide a PFR target, so we have to link to the headers
            # target
            target_link_libraries(kamping_pfr INTERFACE Boost::headers)
            add_library(Boost::pfr ALIAS kamping_pfr)
        endif ()
    else ()
        FetchContent_MakeAvailable(pfr)
        if (pr_FOUND)
            message(STATUS "Found Boost.PFR: ${pfr_DIR}")
        else ()
            message(STATUS "Boost.PFR: building from source.")
        endif ()
    endif ()
    target_link_libraries(kamping_base INTERFACE Boost::pfr)
    target_compile_definitions(kamping_base INTERFACE KAMPING_ENABLE_REFLECTION)
    message(STATUS "Reflection: enabled")
else ()
    message(STATUS "Reflection: disabled")
endif ()

if (KAMPING_ENABLE_SERIALIZATION)
    FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal
        GIT_TAG v1.3.2
        SYSTEM FIND_PACKAGE_ARGS 1.3.2
    )
    set(JUST_INSTALL_CEREAL ON)
    FetchContent_MakeAvailable(cereal)
    if (cereal_FOUND)
        message(STATUS "Found cereal: ${cereal_DIR}")
    else ()
        message(STATUS "Cereal: building from source.")
    endif ()
    target_link_libraries(kamping_base INTERFACE cereal::cereal)
    target_compile_definitions(kamping_base INTERFACE KAMPING_ENABLE_SERIALIZATION)
    message(STATUS "Serialization: enabled")
else ()
    message(STATUS "Serialization: disabled")
endif ()

add_library(kamping::kamping ALIAS kamping)

if (KAMPING_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()
if (KAMPING_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()
