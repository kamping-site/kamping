// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

#include "helpers_for_testing.hpp"

using ::testing::BeginEndDistanceIs;
using ::testing::Each;
using ::testing::ElementsAreArray;

template <typename T>
class OwnContainerTest : public ::testing::Test {
public:
    using value_type = T;
    static T get_non_default_value();
    static T get_default_value() {
        return T{};
    }
};

template <>
inline int OwnContainerTest<int>::get_non_default_value() {
    return 42;
}

template <>
inline double OwnContainerTest<double>::get_non_default_value() {
    return 3.14;
}

template <>
inline bool OwnContainerTest<bool>::get_non_default_value() {
    return true;
}

template <>
inline std::tuple<int, double> OwnContainerTest<std::tuple<int, double>>::get_non_default_value() {
    return {42, 3.14};
}

using MyTypes = ::testing::Types<int, double, bool, std::tuple<int, double>>;
TYPED_TEST_SUITE(OwnContainerTest, MyTypes, );

TYPED_TEST(OwnContainerTest, create_empty) {
    testing::OwnContainer<typename TestFixture::value_type> container;
    EXPECT_EQ(container.size(), 0);
    EXPECT_THAT(container, BeginEndDistanceIs(0));
    EXPECT_EQ(container.copy_count(), 0);
}

TYPED_TEST(OwnContainerTest, create_default_initialized) {
    testing::OwnContainer<typename TestFixture::value_type> container(10);

    auto default_value = TestFixture::get_default_value();

    EXPECT_EQ(container.size(), 10);
    EXPECT_THAT(container, BeginEndDistanceIs(10));

    EXPECT_THAT(container, Each(default_value));

    // test [] operator
    for (size_t i = 0; i < container.size(); i++) {
        EXPECT_EQ(container[i], default_value);
    }
    EXPECT_EQ(container.copy_count(), 0);
}

TYPED_TEST(OwnContainerTest, create_non_default_initialized) {
    auto value = TestFixture::get_non_default_value();

    testing::OwnContainer<typename TestFixture::value_type> container(10, value);

    EXPECT_EQ(container.size(), 10);
    EXPECT_THAT(container, BeginEndDistanceIs(10));

    EXPECT_THAT(container, Each(value));

    // test [] operator
    for (size_t i = 0; i < container.size(); i++) {
        EXPECT_EQ(container[i], value);
    }
    EXPECT_EQ(container.copy_count(), 0);
}

TYPED_TEST(OwnContainerTest, create_initializer_list) {
    testing::OwnContainer<typename TestFixture::value_type> container(
        {TestFixture::get_default_value(), TestFixture::get_non_default_value()}
    );

    EXPECT_EQ(container.size(), 2);
    EXPECT_THAT(container, BeginEndDistanceIs(2));

    std::vector expected{TestFixture::get_default_value(), TestFixture::get_non_default_value()};
    EXPECT_THAT(container, ElementsAreArray(expected));

    EXPECT_EQ(container[0], TestFixture::get_default_value());
    EXPECT_EQ(container[1], TestFixture::get_non_default_value());

    EXPECT_EQ(container.copy_count(), 0);
}

TYPED_TEST(OwnContainerTest, modify_container) {
    auto value = TestFixture::get_non_default_value();

    testing::OwnContainer<typename TestFixture::value_type> container(10);
    container[3] = value;

    std::vector<typename TestFixture::value_type> vec(10);
    vec[3] = value;

    EXPECT_EQ(container.size(), 10);
    EXPECT_THAT(container, BeginEndDistanceIs(10));

    EXPECT_THAT(container, ElementsAreArray(vec));

    // test [] operator
    for (size_t i = 0; i < container.size(); i++) {
        if (i == 3) {
            EXPECT_EQ(container[i], value);
        } else {
            EXPECT_EQ(container[i], TestFixture::get_default_value());
        }
    }
    EXPECT_EQ(container.copy_count(), 0);
}

TYPED_TEST(OwnContainerTest, resize_container) {
    auto value = TestFixture::get_non_default_value();

    testing::OwnContainer<typename TestFixture::value_type> container(10, value);
    container.resize(15);

    std::vector<typename TestFixture::value_type> vec(10, value);
    vec.resize(15);

    EXPECT_EQ(container.size(), 15);
    EXPECT_THAT(container, BeginEndDistanceIs(15));
    EXPECT_THAT(container, ElementsAreArray(vec));
    EXPECT_EQ(container.copy_count(), 0);

    container.resize(5);
    vec.resize(5);

    EXPECT_EQ(container.size(), 5);
    EXPECT_THAT(container, BeginEndDistanceIs(5));
    EXPECT_THAT(container, ElementsAreArray(vec));

    EXPECT_EQ(container.copy_count(), 0);
}

TYPED_TEST(OwnContainerTest, data_works) {
    auto value = TestFixture::get_non_default_value();
    {
        testing::OwnContainer<typename TestFixture::value_type> container(10, value);
        EXPECT_EQ(container.data(), container.begin());
        EXPECT_EQ(container.data(), &container[0]);
        EXPECT_EQ(container.copy_count(), 0);
    }
    {
        testing::OwnContainer<typename TestFixture::value_type> const container(10, value);
        EXPECT_EQ(container.data(), container.begin());
        EXPECT_EQ(container.data(), &container[0]);
        EXPECT_EQ(container.copy_count(), 0);
    }
}

TYPED_TEST(OwnContainerTest, copy) {
    auto value = TestFixture::get_non_default_value();

    testing::OwnContainer<typename TestFixture::value_type> container(10);
    for (size_t i = 0; i < container.size(); i++) {
        if (i % 2 == 0) {
            container[i] = value;
        }
    }
    auto container2 = container;
    EXPECT_EQ(container.copy_count(), 1);
    EXPECT_EQ(container2.copy_count(), 1);
    EXPECT_EQ(container, container2);
    container.resize(0);
    EXPECT_NE(container2, container);

    auto container3{container2};
    EXPECT_EQ(container2, container3);
    EXPECT_EQ(container.copy_count(), 2);
    EXPECT_EQ(container2.copy_count(), 2);
    EXPECT_EQ(container3.copy_count(), 2);
    container2.resize(0);
    EXPECT_NE(container2, container3);
}

TYPED_TEST(OwnContainerTest, move) {
    auto value = TestFixture::get_non_default_value();

    testing::OwnContainer<typename TestFixture::value_type> container(10);
    for (size_t i = 0; i < container.size(); i++) {
        if (i % 2 == 0) {
            container[i] = value;
        }
    }

    std::vector<typename TestFixture::value_type> expected(10);
    for (size_t i = 0; i < container.size(); i++) {
        if (i % 2 == 0) {
            expected[i] = value;
        }
    }

    auto container2 = std::move(container);
    EXPECT_EQ(container.copy_count(), 0);
    EXPECT_EQ(container2.copy_count(), 0);
    EXPECT_EQ(container.size(), 0);

    EXPECT_THAT(container2, ElementsAreArray(expected));

    auto container3{std::move(container2)};
    EXPECT_EQ(container2.copy_count(), 0);
    EXPECT_EQ(container3.copy_count(), 0);
    EXPECT_EQ(container2.size(), 0);

    EXPECT_THAT(container3, ElementsAreArray(expected));
}
