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
#include <type_traits>

#include <gtest/gtest.h>

#include "kamping/span.hpp"

using namespace ::kamping;

TEST(SpanTest, test_to_address_plain_pointer) {
    int  x     = 42;
    int* x_ptr = &x;
    EXPECT_EQ(kamping::internal::to_address(x_ptr), x_ptr);

    int a[3] = {42, 34, 27};
    EXPECT_EQ(kamping::internal::to_address(a), a);
}

TEST(SpanTest, test_to_address_smart_pointer) {
    auto x = std::unique_ptr<int>(new int(42));
    EXPECT_EQ(kamping::internal::to_address(x), x.get());
}

// Test our minimal span implementation
TEST(SpanTest, basic_functionality) {
    std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    Span<int> int_span(values.data(), values.size());
    EXPECT_EQ(values.size(), int_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), int_span.size_bytes());
    EXPECT_FALSE(int_span.empty());
    EXPECT_EQ(values.data(), int_span.data());
    EXPECT_EQ(values.data(), &(*int_span.begin()));
    EXPECT_EQ(std::next(int_span.begin(), static_cast<int>(int_span.size())), int_span.end());

    Span<int const> const_int_span = {values.data(), values.size()};
    EXPECT_EQ(values.size(), const_int_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), const_int_span.size_bytes());
    EXPECT_FALSE(const_int_span.empty());
    EXPECT_EQ(values.data(), const_int_span.data());
    EXPECT_EQ(const_int_span.data(), int_span.data());
    EXPECT_EQ(const_int_span.data(), &(*const_int_span.begin()));
    EXPECT_EQ(std::next(const_int_span.begin(), static_cast<int>(const_int_span.size())), const_int_span.end());

#if (defined(__clang__) && __clang_major__ < 15)
    Span<int> int_iterator_span(values.data(), values.data() + values.size());
#else
    Span<int> int_iterator_span(values.begin(), values.end());
#endif
    EXPECT_EQ(values.size(), int_iterator_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), int_iterator_span.size_bytes());
    EXPECT_FALSE(int_iterator_span.empty());
    EXPECT_EQ(values.data(), int_iterator_span.data());
    EXPECT_EQ(int_iterator_span.data(), &(*int_iterator_span.begin()));
    EXPECT_EQ(
        std::next(int_iterator_span.begin(), static_cast<int>(int_iterator_span.size())),
        int_iterator_span.end()
    );

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || __cplusplus < 202002L) // not C++ 20
    // if C++20 is used, we alias Span to std::span, but we cannot use deduction there, because argument deduction is
    // not allowed for alias templates but only for class templates.
    Span int_iterator_span_deducted(values.begin(), values.end());
    EXPECT_EQ(values.size(), int_iterator_span_deducted.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), int_iterator_span_deducted.size_bytes());
    EXPECT_FALSE(int_iterator_span_deducted.empty());
    EXPECT_EQ(values.data(), int_iterator_span_deducted.data());
    EXPECT_EQ(int_iterator_span_deducted.data(), &(*int_iterator_span_deducted.begin()));
    EXPECT_EQ(
        std::next(int_iterator_span_deducted.begin(), static_cast<int>(int_iterator_span_deducted.size())),
        int_iterator_span_deducted.end()
    );
#endif

    Span<int> int_range_span(values);
    EXPECT_EQ(values.size(), int_range_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), int_range_span.size_bytes());
    EXPECT_FALSE(int_range_span.empty());
    EXPECT_EQ(values.data(), int_range_span.data());
    EXPECT_EQ(int_range_span.data(), &(*int_range_span.begin()));
    EXPECT_EQ(std::next(int_range_span.begin(), static_cast<int>(int_range_span.size())), int_range_span.end());

#if ((defined(_MSVC_LANG) && _MSVC_LANG < 202002L) || __cplusplus < 202002L) // not C++ 20
    // if C++20 is used, we alias Span to std::span, but we cannot use deduction there, because argument deduction is
    // not allowed for alias templates but only for class templates.
    Span int_range_span_deducted(values);
    EXPECT_EQ(values.size(), int_range_span_deducted.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), int_range_span_deducted.size_bytes());
    EXPECT_FALSE(int_range_span_deducted.empty());
    EXPECT_EQ(values.data(), int_range_span_deducted.data());
    EXPECT_EQ(int_range_span_deducted.data(), &(*int_range_span_deducted.begin()));
    EXPECT_EQ(
        std::next(int_range_span_deducted.begin(), static_cast<int>(int_range_span_deducted.size())),
        int_range_span_deducted.end()
    );
#endif

    Span<int> empty_span{values.data(), Span<int>::size_type{0}};
    EXPECT_TRUE(empty_span.empty());
    EXPECT_EQ(0, empty_span.size());
    EXPECT_EQ(0, empty_span.size_bytes());
    EXPECT_EQ(values.data(), empty_span.data());
    EXPECT_EQ(empty_span.begin(), empty_span.end());

    Span<int> nullptr_span{};
    EXPECT_TRUE(nullptr_span.empty());
    EXPECT_EQ(0, nullptr_span.size());
    EXPECT_EQ(0, nullptr_span.size_bytes());
    EXPECT_EQ(nullptr, nullptr_span.data());
    EXPECT_EQ(nullptr_span.begin(), nullptr_span.end());

    static_assert(std::is_pointer_v<decltype(int_span.data())>, "Member data() of int_span does not return a pointer.");
    static_assert(
        std::is_pointer_v<decltype(const_int_span.data())>,
        "Member data() of const_int_span does not return a pointer."
    );
    static_assert(
        std::is_pointer_v<decltype(int_iterator_span.data())>,
        "Member data() of int_iterator_span does not return a pointer."
    );
    static_assert(
        std::is_const_v<std::remove_pointer_t<decltype(const_int_span.data())>>,
        "Member data() of const_int_span does not return a pointer pointing to const memory."
    );
    static_assert(
        !std::is_const_v<std::remove_pointer_t<decltype(int_span.data())>>,
        "Member data() of int_span returns a pointer pointing to const memory, but should be non-const."
    );
    static_assert(
        !std::is_const_v<std::remove_pointer_t<decltype(int_iterator_span.data())>>,
        "Member data() of int_iterator_span returns a pointer pointing to const memory, but should be non-const."
    );

    static_assert(
        std::is_same_v<decltype(int_span)::value_type, decltype(values)::value_type>,
        "Member value_type of int_span does not match the element type of the underlying container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::size_type, decltype(values)::size_type>,
        "Member size_type of int_span does not match the element type of the underlying container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::difference_type, decltype(values)::difference_type>,
        "Member difference_type of int_span does not match the element type of the underlying "
        "container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::pointer, decltype(values)::pointer>,
        "Member pointer of int_span does not match the element type of the underlying container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::const_pointer, decltype(values)::const_pointer>,
        "Member const_pointer of int_span does not match the element type of the underlying container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::reference, decltype(values)::reference>,
        "Member reference of int_span does not match the element type of the underlying container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::const_reference, decltype(values)::const_reference>,
        "Member const_reference of int_span does not match the element type of the underlying container."
    );
}

TEST(SpanTest, iterator) {
    std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    Span<int> int_span(values);
    {
        auto it = int_span.begin();
        for (auto value: values) {
            EXPECT_EQ(value, *it);
            ++it;
        }
        EXPECT_EQ(int_span.end(), it);
    }

    // Test reverse iterators
    {
        auto rit = int_span.rbegin();
        for (auto it = values.rbegin(); it != values.rend(); ++it) {
            EXPECT_EQ(*it, *rit);
            ++rit;
        }
        EXPECT_EQ(int_span.rend(), rit);
    }
}

TEST(SpanTest, accessors) {
    std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    Span<int> int_span(values);
    EXPECT_EQ(values.front(), 1);
    EXPECT_EQ(values.back(), 10);
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], int_span[i]);
    }
}

TEST(SpanTest, subspans) {
    std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    Span<int> int_span(values);
    auto      first_5 = int_span.first(5);
    EXPECT_THAT(first_5, testing::ElementsAreArray({1, 2, 3, 4, 5}));

    auto last_5 = int_span.last(5);
    EXPECT_THAT(last_5, testing::ElementsAreArray({6, 7, 8, 9, 10}));

    auto subspan = int_span.subspan(3, 4);
    EXPECT_THAT(subspan, testing::ElementsAreArray({4, 5, 6, 7}));
}
