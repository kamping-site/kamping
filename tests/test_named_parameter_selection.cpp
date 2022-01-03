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

#include <gtest/gtest.h>

#include "kamping/buffer_factories.hpp"
#include "kamping/named_parameter_selection.hpp"

using namespace ::kamping::internal;

// Mock argument for testing the named parameter selection mechanism
template <ParameterType _ptype>
struct Argument {
    static constexpr ParameterType ptype = _ptype;
    Argument(int i) : _i{i} {}
    int _i;
};

TEST(HelpersTest, select_ptype_basics) {
    Argument<ParameterType::send_buf>    arg0{0};
    Argument<ParameterType::recv_buf>    arg1{1};
    Argument<ParameterType::send_counts> arg2{2};
    {
        const auto& selected_arg = select_ptype<ParameterType::send_buf>(arg0, arg1, arg2);
        EXPECT_EQ(selected_arg._i, 0);
    }
    {
        const auto& selected_arg = select_ptype<ParameterType::recv_buf>(arg0, arg1, arg2);
        EXPECT_EQ(selected_arg._i, 1);
    }
    {
        const auto& selected_arg = select_ptype<ParameterType::send_counts>(arg0, arg1, arg2);
        EXPECT_EQ(selected_arg._i, 2);
    }
}

TEST(HelpersTest, select_ptype_duplicates) {
    Argument<ParameterType::send_buf>    arg0{0};
    Argument<ParameterType::recv_buf>    arg1{1};
    Argument<ParameterType::send_counts> arg2{2};
    Argument<ParameterType::send_buf>    arg3{3};
    {
        // if two arguments have the same ParameterType the first occurence in the argument list is selected
        const auto& selected_arg = select_ptype<ParameterType::send_buf>(arg0, arg1, arg2, arg3);
        EXPECT_EQ(selected_arg._i, 0);
    }
}
