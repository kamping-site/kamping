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

#include "kamping/mpi_ops.hpp"
#include "kamping/named_parameters.hpp"

using namespace ::kamping;
using namespace ::kamping::internal;

TEST(ParameterFactoriesTest, op_commutativity_tags_work) {
    struct MySum {
        int operator()(int const& a, int const& b) const {
            return a + b;
        }
    };
    {
        auto op_object = op(std::plus<>{});
        auto op        = op_object.build_operation<int>();
        EXPECT_EQ(op.op(), MPI_SUM);
        EXPECT_TRUE(decltype(op)::commutative);
    }
    {
        auto op_object = op(MySum{}, kamping::ops::commutative);
        auto op        = op_object.build_operation<int>();
        EXPECT_NE(op.op(), MPI_SUM);
        EXPECT_TRUE(decltype(op)::commutative);
    }
    {
        auto op_object = op(MySum{}, kamping::ops::non_commutative);
        auto op        = op_object.build_operation<int>();
        EXPECT_NE(op.op(), MPI_SUM);
        EXPECT_FALSE(decltype(op)::commutative);
    }
}
