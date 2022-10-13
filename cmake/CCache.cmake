# Adopted from: https://crascit.com/2016/04/09/using-ccache-with-cmake/#more-410

set(CCACHE_LAUNCHER_DIR "${CMAKE_CURRENT_LIST_DIR}")

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    message(STATUS "Using ccache to speed up compilation.")

    set(SUPPORTED_GENERATORS "Unix Makefiles" "Ninja")
    if (CMAKE_GENERATOR IN_LIST SUPPORTED_GENERATORS)
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    else ()
        message(WARNING "${CMAKE_GENERATOR} does not support <LANG>_COMPILER_LAUNCHER.")
    endif ()
else ()
    message(WARNING "Could not find ccache in your PATH.")
endif ()
