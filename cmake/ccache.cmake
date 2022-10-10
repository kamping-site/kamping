# Adopted from: https://crascit.com/2016/04/09/using-ccache-with-cmake/#more-410

set(CCACHE_LAUNCHER_DIR "${CMAKE_CURRENT_LIST_DIR}")

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    message(STATUS "Using ccache to speed up compilation.")

    # Set up wrapper scripts
    set(C_LAUNCHER "${CCACHE_PROGRAM}")
    set(CXX_LAUNCHER "${CCACHE_PROGRAM}")
    configure_file("${CCACHE_LAUNCHER_DIR}/launch-c.in" launch-c)
    configure_file("${CCACHE_LAUNCHER_DIR}/launch-cxx.in" launch-cxx)
    execute_process(COMMAND chmod a+rx "${CMAKE_BINARY_DIR}/launch-c" "${CMAKE_BINARY_DIR}/launch-cxx")

    if (CMAKE_GENERATOR STREQUAL "Xcode")
        # Set Xcode project attributes to route compilation and linking through our scripts
        set(CMAKE_XCODE_ATTRIBUTE_CC "${CMAKE_BINARY_DIR}/launch-c")
        set(CMAKE_XCODE_ATTRIBUTE_CXX "${CMAKE_BINARY_DIR}/launch-cxx")
        set(CMAKE_XCODE_ATTRIBUTE_LD "${CMAKE_BINARY_DIR}/launch-c")
        set(CMAKE_XCODE_ATTRIBUTE_LDPLUSPLUS "${CMAKE_BINARY_DIR}/launch-cxx")
    else ()
        # Support Unix Makefiles and Ninja
        set(CMAKE_C_COMPILER_LAUNCHER "${CMAKE_BINARY_DIR}/launch-c")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CMAKE_BINARY_DIR}/launch-cxx")
    endif ()
endif ()
