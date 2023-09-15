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

#include "helpers_for_testing.hpp"
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
template <typename ExpectedValueType, typename GeneratedBuffer>
void test_single_element_buffer(
    GeneratedBuffer const&           generatedbuffer,
    kamping::internal::ParameterType expected_parameter_type,
    kamping::internal::BufferType    expected_buffer_type,
    ExpectedValueType const          value,
    bool                             should_be_modifiable = false
) {
    static_assert(std::is_same_v<typename GeneratedBuffer::value_type, ExpectedValueType>);

    EXPECT_EQ(GeneratedBuffer::is_modifiable, should_be_modifiable);
    EXPECT_EQ(GeneratedBuffer::parameter_type, expected_parameter_type);
    EXPECT_EQ(GeneratedBuffer::buffer_type, expected_buffer_type);

    auto get_result = generatedbuffer.get();
    EXPECT_EQ(get_result.size(), 1);
    EXPECT_EQ(*(get_result.data()), value);
}

TEST(ParameterFactoriesTest, send_type_custom_type) {
    MPI_Datatype custom_type = testing::MPI_INT_padding_MPI_INT();
    MPI_Type_commit(&custom_type);
    auto send_type          = kamping::send_type(custom_type);
    using ExpectedValueType = MPI_Datatype;
    test_single_element_buffer<ExpectedValueType>(
        send_type,
        ParameterType::send_type,
        internal::BufferType::in_buffer,
        custom_type
    );
    MPI_Type_free(&custom_type);
}

TEST(ParameterFactoriesTest, recv_type_custom_type) {
    MPI_Datatype custom_type = testing::MPI_INT_padding_MPI_INT();
    MPI_Type_commit(&custom_type);
    auto recv_type          = kamping::recv_type(custom_type);
    using ExpectedValueType = MPI_Datatype;
    test_single_element_buffer<ExpectedValueType>(
        recv_type,
        ParameterType::recv_type,
        internal::BufferType::in_buffer,
        custom_type
    );
    MPI_Type_free(&custom_type);
}
