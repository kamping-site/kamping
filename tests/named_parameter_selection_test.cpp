// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <gtest/gtest.h>
#include <tuple>

#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"

#include "helpers_for_testing.hpp"

using namespace ::kamping::internal;

TEST(NamedParameterTest, select_parameter_type_basics) {
    testing::Argument<ParameterType::send_buf>    arg0{0};
    testing::Argument<ParameterType::recv_buf>    arg1{1};
    testing::Argument<ParameterType::send_counts> arg2{2};
    {
        const auto& selected_arg = select_parameter_type<ParameterType::send_buf>(arg0, arg1, arg2);
        EXPECT_EQ(selected_arg._i, 0);
    }
    {
        const auto& selected_arg = select_parameter_type<ParameterType::recv_buf>(arg0, arg1, arg2);
        EXPECT_EQ(selected_arg._i, 1);
    }
    {
        const auto& selected_arg = select_parameter_type<ParameterType::send_counts>(arg0, arg1, arg2);
        EXPECT_EQ(selected_arg._i, 2);
    }
}

TEST(NamedParameterTest, has_parameter_type_basics) {
    testing::Argument<ParameterType::send_buf>    arg0{0};
    testing::Argument<ParameterType::recv_buf>    arg1{1};
    testing::Argument<ParameterType::send_counts> arg2{2};

    EXPECT_TRUE(has_parameter_type<ParameterType::send_buf>(arg0, arg1, arg2));
    EXPECT_TRUE(has_parameter_type<ParameterType::recv_buf>(arg0, arg1, arg2));
    EXPECT_TRUE(has_parameter_type<ParameterType::send_counts>(arg0, arg1, arg2));
    EXPECT_FALSE(has_parameter_type<ParameterType::root>(arg0, arg1, arg2));
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
            std::tuple(42), arg0, arg1, arg2);
        static_assert(std::is_same_v<decltype(selected_arg), decltype(arg0)&>);
        EXPECT_EQ(selected_arg._i, 0);
    }
    {
        auto&& selected_arg = select_parameter_type_or_default<ParameterType::recv_buf, DefaultArgument>(
            std::tuple(42), arg0, arg1, arg2);
        static_assert(std::is_same_v<decltype(selected_arg), decltype(arg1)&>);
        EXPECT_EQ(selected_arg._i, 1);
    }
    {
        auto&& selected_arg = select_parameter_type_or_default<ParameterType::send_counts, DefaultArgument>(
            std::tuple(42), arg0, arg1, arg2);
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
            std::tuple(42, "KaMPI.ng"), arg0, arg1, arg2);
        static_assert(std::is_same_v<decltype(selected_arg), DefaultArgument&&>);
        EXPECT_EQ(selected_arg._value, 42);
        EXPECT_EQ(selected_arg._message, "KaMPI.ng");
    }
}

TEST(NamedParameterTest, select_parameter_type_duplicates) {
    testing::Argument<ParameterType::send_buf>    arg0{0};
    testing::Argument<ParameterType::recv_buf>    arg1{1};
    testing::Argument<ParameterType::send_counts> arg2{2};
    testing::Argument<ParameterType::send_buf>    arg3{3};
    {
        // if two arguments have the same ParameterType the first occurence in the argument list is selected
        const auto& selected_arg = select_parameter_type<ParameterType::send_buf>(arg0, arg1, arg2, arg3);
        EXPECT_EQ(selected_arg._i, 0);
    }
}

// @brief This dummy ressembles the interface of a collective operation, so we can simulate the check for rvalue
// parameters.
template <typename... Args>
bool dummy_collective_operation(Args&&... args [[maybe_unused]]) {
    return all_parameters_are_rvalues<Args...>;
}

TEST(NamedParameterTest, all_parameters_are_rvalues) {
    testing::Argument<ParameterType::send_buf> arg0{0};
    testing::Argument<ParameterType::recv_buf> arg1{1};
    {
        EXPECT_FALSE(dummy_collective_operation(arg0, arg1));
        EXPECT_FALSE(dummy_collective_operation(decltype(arg0){0}, arg1));
        EXPECT_FALSE(dummy_collective_operation(arg0, decltype(arg1){1}));
        EXPECT_TRUE(dummy_collective_operation(decltype(arg0){0}, decltype(arg1){1}));
    }
}
