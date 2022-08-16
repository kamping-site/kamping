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

#include <type_traits>

#include <gtest/gtest.h>

#include "kamping/span.hpp"

using namespace ::kamping;

// Test our minimal span implementation
TEST(SpanTest, basic_functionality) {
    std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    Span<int> int_span(values.data(), values.size());
    EXPECT_EQ(values.size(), int_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), int_span.size_bytes());
    EXPECT_FALSE(int_span.empty());
    EXPECT_EQ(values.data(), int_span.data());

    Span<int> tuple_constructed_span(std::tuple<int*, size_t>{values.data(), values.size()});
    EXPECT_EQ(values.size(), tuple_constructed_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), tuple_constructed_span.size_bytes());
    EXPECT_FALSE(tuple_constructed_span.empty());
    EXPECT_EQ(values.data(), tuple_constructed_span.data());
    EXPECT_EQ(tuple_constructed_span.data(), int_span.data());

    Span<int const> const_int_span = {values.data(), values.size()};
    EXPECT_EQ(values.size(), const_int_span.size());
    EXPECT_EQ(values.size() * sizeof(decltype(values)::value_type), const_int_span.size_bytes());
    EXPECT_FALSE(const_int_span.empty());
    EXPECT_EQ(values.data(), const_int_span.data());
    EXPECT_EQ(const_int_span.data(), int_span.data());

    Span<int> empty_span = {values.data(), 0};
    EXPECT_TRUE(empty_span.empty());
    EXPECT_EQ(0, empty_span.size());
    EXPECT_EQ(0, empty_span.size_bytes());
    EXPECT_EQ(values.data(), empty_span.data());

    Span<int> nullptr_span = {nullptr, 0};
    EXPECT_TRUE(nullptr_span.empty());
    EXPECT_EQ(0, nullptr_span.size());
    EXPECT_EQ(0, nullptr_span.size_bytes());
    EXPECT_EQ(nullptr, nullptr_span.data());

    static_assert(
        std::is_pointer_v<decltype(int_span.data())>,
        "Member data() of internal::Span<T*, size_t> does not return a pointer."
    );
    static_assert(
        std::is_pointer_v<decltype(const_int_span.data())>,
        "Member data() of internal::Span<T const *, size_t> does not return a pointer."
    );
    static_assert(
        std::is_const_v<std::remove_pointer_t<decltype(const_int_span.data())>>,
        "Member data() of internal::Span<T const *, size_t> does not return a pointer pointing to const memory."
    );
    static_assert(
        !std::is_const_v<std::remove_pointer_t<decltype(int_span.data())>>,
        "Member data() of internal::Span<T*, size_t> does return a pointer pointing to const memory."
    );

    static_assert(
        std::is_same_v<decltype(int_span)::value_type, decltype(values)::value_type>,
        "Member value_type of internal::Span<T*, size_t> does not match the element type of the underlying container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::size_type, decltype(values)::size_type>,
        "Member size_type of internal::Span<T*, size_t> does not match the element type of the underlying container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::difference_type, decltype(values)::difference_type>,
        "Member difference_type of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::pointer, decltype(values)::pointer>,
        "Member pointer of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::const_pointer, decltype(values)::const_pointer>,
        "Member const_pointer of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::reference, decltype(values)::reference>,
        "Member reference of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container."
    );
    static_assert(
        std::is_same_v<decltype(int_span)::const_reference, decltype(values)::const_reference>,
        "Member const_reference of internal::Span<T*, difference_t> does not match the element type of the underlying "
        "container."
    );
}
