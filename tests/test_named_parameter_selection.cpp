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

#include "kamping/named_parameter_selection.hpp"
#include "kamping/parameter_factories.hpp"

#include "helpers_for_testing.hpp"

using namespace ::kamping::internal;

TEST(HelpersTest, select_parameter_type_basics) {
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

TEST(HelpersTest, select_parameter_type_duplicates) {
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
