#include <gtest/gtest-death-test.h>

///
/// @brief Verifies that the expression passed does not crash
///
/// This works exiting with a 0 code after the expression and letting gtest check whether that exit occured.
/// From https://stackoverflow.com/questions/60594487/expect-no-death-in-google-test
///
/// @param expression The expression that should succeed
///
#define EXPECT_NO_DEATH(expression)          \
    EXPECT_EXIT(                             \
        {                                    \
            { #expression; }                 \
            fprintf(stderr, "Still alive!"); \
            exit(0);                         \
        },                                   \
        ::testing::ExitedWithCode(0), "Still alive!");
