#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>

#include "kamping/helpers.hpp"

using namespace ::testing;

TEST(HelpersTest, in_range) {
    uint8_t u8val = 200;
    ASSERT_TRUE(in_range<uint8_t>(u8val));
    ASSERT_TRUE(in_range<uint16_t>(u8val));
    ASSERT_TRUE(in_range<uint32_t>(u8val));
    ASSERT_TRUE(in_range<uint64_t>(u8val));
    ASSERT_FALSE(in_range<int8_t>(u8val));
    ASSERT_TRUE(in_range<int16_t>(u8val));
    ASSERT_TRUE(in_range<int32_t>(u8val));
    ASSERT_TRUE(in_range<int64_t>(u8val));
    u8val = 10;
    ASSERT_TRUE(in_range<int8_t>(u8val));

    auto intMax = std::numeric_limits<int>::max();
    ASSERT_TRUE(in_range<long int>(intMax));
    ASSERT_TRUE(in_range<uintmax_t>(intMax));
    ASSERT_TRUE(in_range<intmax_t>(intMax));

    auto intNeg = -1;
    ASSERT_TRUE(in_range<long int>(intNeg));
    ASSERT_FALSE(in_range<uintmax_t>(intNeg));
    ASSERT_TRUE(in_range<intmax_t>(intNeg));
    ASSERT_FALSE(in_range<size_t>(intNeg));
    ASSERT_TRUE(in_range<short int>(intNeg));

    size_t sizeT = 10000;
    ASSERT_TRUE(in_range<int>(sizeT));
    sizeT = std::numeric_limits<size_t>::max() - 1000;
    ASSERT_FALSE(in_range<int>(sizeT));
    ASSERT_TRUE(in_range<uintmax_t>(sizeT));

    unsigned long a = 16;
    ASSERT_TRUE(in_range<unsigned char>(a));

    // Cast large values into narrower types.
    ASSERT_FALSE(in_range<uint8_t>(std::numeric_limits<uint16_t>::max()));
    ASSERT_FALSE(in_range<uint16_t>(std::numeric_limits<uint32_t>::max() - 1000));
    ASSERT_FALSE(in_range<uint32_t>(std::numeric_limits<uint64_t>::max() - 133742));

    ASSERT_FALSE(in_range<int8_t>(std::numeric_limits<int16_t>::max()));
    ASSERT_FALSE(in_range<int8_t>(std::numeric_limits<int16_t>::min()));
    ASSERT_FALSE(in_range<int16_t>(std::numeric_limits<int32_t>::max()));
    ASSERT_FALSE(in_range<int16_t>(std::numeric_limits<int32_t>::min()));
    ASSERT_FALSE(in_range<int32_t>(std::numeric_limits<int64_t>::max()));
    ASSERT_FALSE(in_range<int32_t>(std::numeric_limits<int64_t>::min()));
}

TEST(HelpersTest, asserting_cast) {
    uint8_t u8val = 200;
    // There is no ASSERT_NO_DEATH()
    ASSERT_EQ(asserting_cast<uint8_t>(u8val), 200);

    // According to the googletest documentation, throwing an exception is not considered a death.
    ASSERT_DEATH(asserting_cast<int8_t>(u8val), "Assertion `in_range<To>\\(value\\)' failed.");
}

TEST(HelpersTest, throwing_cast) {
    uint8_t u8val = 200;
    ASSERT_NO_THROW(throwing_cast<uint8_t>(u8val));
    ASSERT_THROW(throwing_cast<int8_t>(u8val), std::range_error);

    // Check the error messages.
    try {
        throwing_cast<int8_t>(1337);
    } catch (std::exception& e) {
        ASSERT_EQ(e.what(), std::string("1337 is not not representable the target type."));
    }

    // ... for negative values.
    try {
        throwing_cast<uint8_t>(-42);
    } catch (std::exception& e) {
        ASSERT_EQ(e.what(), std::string("-42 is not not representable the target type."));
    }
}