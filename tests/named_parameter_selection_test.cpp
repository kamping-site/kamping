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

#include <tuple>

#include <gtest/gtest.h>

#include "helpers_for_testing.hpp"
#include "kamping/named_parameter_selection.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping::internal;

TEST(NamedParameterTest, select_parameter_type_basics) {
    testing::Argument<ParameterType::send_buf>    arg0{0};
    testing::Argument<ParameterType::recv_buf>    arg1{1};
    testing::Argument<ParameterType::send_counts> arg2{2};
    {
        auto const& selected_arg = select_parameter_type<ParameterType::send_buf>(arg0, arg1, arg2);
        EXPECT_EQ(selected_arg._i, 0);
    }
    {
        auto const& selected_arg = select_parameter_type<ParameterType::recv_buf>(arg0, arg1, arg2);
        EXPECT_EQ(selected_arg._i, 1);
    }
    {
        auto const& selected_arg = select_parameter_type<ParameterType::send_counts>(arg0, arg1, arg2);
        EXPECT_EQ(selected_arg._i, 2);
    }
}

TEST(NamedParameterTest, has_parameter_type_basics_compile_time) {
    testing::Argument<ParameterType::send_buf>    arg0{0};
    testing::Argument<ParameterType::recv_buf>    arg1{1};
    testing::Argument<ParameterType::send_counts> arg2{2};

    static_assert(has_parameter_type<ParameterType::send_buf, decltype(arg0), decltype(arg1), decltype(arg2)>());
    static_assert(has_parameter_type<ParameterType::recv_buf, decltype(arg0), decltype(arg1), decltype(arg2)>());
    static_assert(has_parameter_type<ParameterType::send_counts, decltype(arg0), decltype(arg1), decltype(arg2)>());
    static_assert(!has_parameter_type<ParameterType::root, decltype(arg0), decltype(arg1), decltype(arg2)>());
}

TEST(NamedParameterTest, default_parameters) {
    struct DefaultArgument {
        DefaultArgument(int value, std::string message = "Hello") : _value(value), _message(message) {}
        int         _value;
        std::string _message;
    };
    testing::Argument<ParameterType::send_buf>    arg0{0};
    testing::Argument<ParameterType::recv_buf>    arg1{1};
    testing::Argument<ParameterType::send_counts> arg2{2};

    {
        auto&& selected_arg = select_parameter_type_or_default<ParameterType::send_buf, DefaultArgument>(
            std::tuple(42),
            arg0,
            arg1,
            arg2
        );
        static_assert(std::is_same_v<decltype(selected_arg), decltype(arg0)&>);
        EXPECT_EQ(selected_arg._i, 0);
    }
    {
        auto&& selected_arg = select_parameter_type_or_default<ParameterType::recv_buf, DefaultArgument>(
            std::tuple(42),
            arg0,
            arg1,
            arg2
        );
        static_assert(std::is_same_v<decltype(selected_arg), decltype(arg1)&>);
        EXPECT_EQ(selected_arg._i, 1);
    }
    {
        auto&& selected_arg = select_parameter_type_or_default<ParameterType::send_counts, DefaultArgument>(
            std::tuple(42),
            arg0,
            arg1,
            arg2
        );
        static_assert(std::is_same_v<decltype(selected_arg), decltype(arg2)&>);
        EXPECT_EQ(selected_arg._i, 2);
    }
    {
        auto&& selected_arg =
            select_parameter_type_or_default<ParameterType::root, DefaultArgument>(std::tuple(42), arg0, arg1, arg2);
        static_assert(std::is_same_v<decltype(selected_arg), DefaultArgument&&>);
        EXPECT_EQ(selected_arg._value, 42);
        EXPECT_EQ(selected_arg._message, "Hello");
    }
    {
        auto&& selected_arg = select_parameter_type_or_default<ParameterType::root, DefaultArgument>(
            std::tuple(42, "KaMPIng"),
            arg0,
            arg1,
            arg2
        );
        static_assert(std::is_same_v<decltype(selected_arg), DefaultArgument&&>);
        EXPECT_EQ(selected_arg._value, 42);
        EXPECT_EQ(selected_arg._message, "KaMPIng");
    }
}

TEST(NamedParameterTest, select_parameter_type_duplicates) {
    testing::Argument<ParameterType::send_buf>    arg0{0};
    testing::Argument<ParameterType::recv_buf>    arg1{1};
    testing::Argument<ParameterType::send_counts> arg2{2};
    testing::Argument<ParameterType::send_buf>    arg3{3};
    {
        // If two arguments have the same ParameterType the first occurrence in the argument list is selected.
        auto const& selected_arg = select_parameter_type<ParameterType::send_buf>(arg0, arg1, arg2, arg3);
        EXPECT_EQ(selected_arg._i, 0);
    }
}

// Test that has_parameter_type can be invoked if the function is called with zero arguments
template <typename... Args>
bool dummy_test_has_parameter(Args... args [[maybe_unused]]) {
    return has_parameter_type<ParameterType::send_buf, Args...>();
}

TEST(NamedParameterTest, has_parameter_on_empty_args) {
    EXPECT_FALSE(dummy_test_has_parameter());
}
