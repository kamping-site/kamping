include(FetchContent)

FetchContent_Declare(
    Format.cmake
    GIT_REPOSITORY https://github.com/TheLartians/Format.cmake
    GIT_TAG v1.8.1
)
FetchContent_MakeAvailable(Format.cmake)
