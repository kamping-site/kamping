// This file is part of KaMPIng.
//
// Copyright 2021 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>
#include <kassert/kassert.hpp>

#include "kamping/assertion_levels.hpp"
#include "kamping/checking_casts.hpp"

using namespace ::testing;
using namespace ::kamping;

TEST(CheckingCastTest, in_range) {
    uint8_t u8val = 200;
    EXPECT_TRUE(in_range<uint8_t>(u8val));
    EXPECT_TRUE(in_range<uint16_t>(u8val));
    EXPECT_TRUE(in_range<uint32_t>(u8val));
    EXPECT_TRUE(in_range<uint64_t>(u8val));
    EXPECT_FALSE(in_range<int8_t>(u8val));
    EXPECT_TRUE(in_range<int16_t>(u8val));
    EXPECT_TRUE(in_range<int32_t>(u8val));
    EXPECT_TRUE(in_range<int64_t>(u8val));
    u8val = 10;
    EXPECT_TRUE(in_range<int8_t>(u8val));

    auto intMax = std::numeric_limits<int>::max();
    EXPECT_TRUE(in_range<long int>(intMax));
    EXPECT_TRUE(in_range<uintmax_t>(intMax));
    EXPECT_TRUE(in_range<intmax_t>(intMax));

    auto intNeg = -1;
    EXPECT_TRUE(in_range<long int>(intNeg));
    EXPECT_FALSE(in_range<uintmax_t>(intNeg));
    EXPECT_TRUE(in_range<intmax_t>(intNeg));
    EXPECT_FALSE(in_range<size_t>(intNeg));
    EXPECT_TRUE(in_range<short int>(intNeg));

    size_t sizeT = 10000;
    EXPECT_TRUE(in_range<int>(sizeT));
    sizeT = std::numeric_limits<size_t>::max() - 1000;
    EXPECT_FALSE(in_range<int>(sizeT));
    EXPECT_TRUE(in_range<uintmax_t>(sizeT));

    unsigned long a = 16;
    EXPECT_TRUE(in_range<unsigned char>(a));

    // Cast large values into narrower types.
    EXPECT_FALSE(in_range<uint8_t>(std::numeric_limits<uint16_t>::max()));
    EXPECT_FALSE(in_range<uint16_t>(std::numeric_limits<uint32_t>::max() - 1000));
    EXPECT_FALSE(in_range<uint32_t>(std::numeric_limits<uint64_t>::max() - 133742));

    EXPECT_FALSE(in_range<int8_t>(std::numeric_limits<int16_t>::max()));
    EXPECT_FALSE(in_range<int8_t>(std::numeric_limits<int16_t>::min()));
    EXPECT_FALSE(in_range<int16_t>(std::numeric_limits<int32_t>::max()));
    EXPECT_FALSE(in_range<int16_t>(std::numeric_limits<int32_t>::min()));
    EXPECT_FALSE(in_range<int32_t>(std::numeric_limits<int64_t>::max()));
    EXPECT_FALSE(in_range<int32_t>(std::numeric_limits<int64_t>::min()));
}

TEST(CheckingCastTest, asserting_cast) {
    uint8_t u8val = 200;

    // Verify that asserting_cast does not crash
    // This works by exiting with a 0 code after the expression and letting gtest check whether that exit occurred.
    // From https://stackoverflow.com/questions/60594487/expect-no-death-in-google-test
    EXPECT_EXIT(
        {
            asserting_cast<uint8_t>(u8val);
            fprintf(stderr, "Still alive!");
            exit(0);
        },
        ::testing::ExitedWithCode(0),
        "Still alive"
    );

    if constexpr (KASSERT_ASSERTION_LEVEL >= kamping::assert::normal) {
        // According to the googletest documentation, throwing an exception is not considered a death.
        // This ASSERT should therefore only succeed if an assert() fails, not if an exception is thrown.
        EXPECT_DEATH(asserting_cast<int8_t>(u8val), "FAILED ASSERTION");
    } else {
        EXPECT_EXIT(
            {
                asserting_cast<int8_t>(u8val);
                fprintf(stderr, "Still alive!");
                exit(0);
            },
            ::testing::ExitedWithCode(0),
            "Still alive"
        );
    }
}

///
/// @brief Checks if a functions fails with a std::range_error exception if exception mode is enabled and an assertion
/// when exception mode is disabled using google test.
///
/// @tparam Lambda Function to check for failures.
/// @param callable Function to check for failures.
/// @param what Substring that should be contained in the output of what() of the thrown exception. Ignored if empty.
///
template <typename Lambda>
void checkThrowOrAssert(Lambda&& callable, [[maybe_unused]] std::string const& what = std::string()) {
#if KASSERT_EXCEPTION_MODE == 0
    if constexpr (KASSERT_ASSERTION_LEVEL >= kassert::assert::kthrow) {
        EXPECT_DEATH(callable(), "FAILED");
    } else {
        EXPECT_EXIT(
            {
                callable();
                fprintf(stderr, "Still alive!");
                exit(0);
            },
            ::testing::ExitedWithCode(0),
            "Still alive"
        );
    }
#else
    EXPECT_THROW(callable(), std::range_error);
    if (!what.empty()) {
        try {
            callable();
        } catch (std::exception& e) {
            EXPECT_THAT(e.what(), HasSubstr(what));
        }
    }
#endif
}

TEST(CheckingCastTest, throwing_cast) {
    uint8_t u8val = 200;

    // A valid cast does not throw an exception.
    EXPECT_NO_THROW(throwing_cast<uint8_t>(u8val));

    // An invalid cast throws an exception.
    checkThrowOrAssert([&]() { return throwing_cast<int8_t>(u8val); });

    // Check the error messages.
    checkThrowOrAssert([&]() { return throwing_cast<int8_t>(1337); }, "1337 is not representable by the target type.");

    // ... for negative values.
    checkThrowOrAssert([&]() { return throwing_cast<uint8_t>(-42); }, "-42 is not representable by the target type.");
}
