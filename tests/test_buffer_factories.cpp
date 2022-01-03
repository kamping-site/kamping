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

  using namespace ::kamping;

TEST(HelpersTest, send_buf_basics) {
    std::vector<int>       int_vec{1, 2, 3, 4, 5, 6};
    std::vector<int> const int_vec_const{1, 2, 3, 4, 5, 6};

    auto gen_via_int_vec       = send_buf(int_vec);
    auto gen_via_int_vec_const = send_buf(int_vec_const);

    // value_type of a buffer should be the same as the value_type of the underlying container
    static_assert(std::is_same_v<typename decltype(gen_via_int_vec)::value_type, int>);
    static_assert(std::is_same_v<typename decltype(gen_via_int_vec_const)::value_type, int>);

    // send buffers are always read-only as the do not need to be modified
    EXPECT_FALSE(decltype(gen_via_int_vec)::is_modifiable);
    EXPECT_FALSE(decltype(gen_via_int_vec_const)::is_modifiable);

    EXPECT_EQ(decltype(gen_via_int_vec)::ptype, kamping::internal::ParameterType::send_buf);
    EXPECT_EQ(decltype(gen_via_int_vec_const)::ptype, kamping::internal::ParameterType::send_buf);

    const auto span_int_vec       = gen_via_int_vec.get();
    const auto span_int_vec_const = gen_via_int_vec_const.get();

    EXPECT_EQ(span_int_vec.ptr, int_vec.data());
    EXPECT_EQ(span_int_vec_const.ptr, int_vec_const.data());
    EXPECT_EQ(span_int_vec_const.size, 6);
    EXPECT_EQ(span_int_vec_const.size, 6);
}
