
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

#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(AllreduceTest, reduce_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto result = comm.allreduce(send_buf(input), op(kamping::ops::plus<>{})).extract_recv_buffer();
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, allreduce_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.allreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf(result));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, allreduce_builtin_op_on_non_builtin_type) {
    Communicator comm;

    struct MyInt {
        MyInt() : _value(0) {}
        MyInt(int value) : _value(value) {}
        int _value;
        int operator+(MyInt const& rhs) const noexcept {
            return this->_value + rhs._value;
        }
        bool operator==(MyInt const& rhs) const {
            return this->_value == rhs._value;
        }
    };
    std::vector<MyInt> input = {comm.rank_signed(), 42};

    auto result =
        comm.allreduce(send_buf(input), op(kamping::ops::plus<>{}, kamping::commutative)).extract_recv_buffer();
    EXPECT_EQ(result.size(), 2);
    std::vector<MyInt> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

int add_plus_42_function(int const& lhs, int const& rhs) {
    return lhs + rhs + 42;
}

TEST(AllreduceTest, allreduce_custom_operation_on_builtin_type) {
    Communicator comm;

    auto add_plus_42_lambda = [](auto const& lhs, auto const& rhs) {
        return lhs + rhs + 42;
    };

    std::vector<int> input = {0, 17, 8};

    { // use function ptr
        auto result =
            comm.allreduce(send_buf(input), op(add_plus_42_function, kamping::commutative)).extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use lambda
        auto result =
            comm.allreduce(send_buf(input), op(add_plus_42_lambda, kamping::commutative)).extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use lambda inline
        auto result = comm.allreduce(
                              send_buf(input),
                              op([](auto const& lhs, auto const& rhs) { return lhs + rhs + 42; }, kamping::commutative))
                          .extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use function object
        struct MySum42 {
            int operator()(int const& lhs, int const& rhs) {
                return lhs + rhs + 42;
            }
        };
        auto result = comm.allreduce(send_buf(input), op(MySum42{}, kamping::commutative)).extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            comm.size_signed() * 0 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 17 + (comm.size_signed() - 1) * 42,
            comm.size_signed() * 8 + (comm.size_signed() - 1) * 42};
        EXPECT_EQ(result, expected_result);
    }
}

TEST(AllreduceTest, allreduce_custom_operation_on_builtin_type_non_commutative) {
    Communicator comm;

    auto get_right = [](auto const&, auto const& rhs) {
        return rhs;
    };

    std::vector<int> input = {comm.rank_signed() + 17};

    auto result = comm.allreduce(send_buf(input), op(get_right, kamping::non_commutative)).extract_recv_buffer();

    EXPECT_EQ(result.size(), 1);
    std::vector<int> expected_result = {comm.size_signed() - 1 + 17};
    EXPECT_EQ(result, expected_result);
}

TEST(AllreduceTest, allreduce_custom_operation_on_custom_type) {
    Communicator comm;

    struct Aggregate {
        int  min;
        int  max;
        bool flag;
        int  sum;

        bool operator==(Aggregate const& rhs) const {
            return this->min == rhs.min && this->max == rhs.max && this->flag == rhs.flag && this->sum == rhs.sum;
        }
    };
    auto my_op = [](Aggregate const& lhs, Aggregate const& rhs) {
        Aggregate agg;
        agg.min  = std::min(lhs.min, rhs.min);
        agg.max  = std::max(lhs.max, rhs.max);
        agg.flag = lhs.flag || rhs.flag;
        agg.sum  = lhs.sum + rhs.sum;
        return agg;
    };

    Aggregate agg1               = {comm.rank_signed(), comm.rank_signed(), false, 1};
    agg1.flag                    = true;
    Aggregate              agg2  = {comm.rank_signed() + 42, comm.rank_signed() + 42, false, 1};
    std::vector<Aggregate> input = {agg1, agg2};

    Aggregate              agg1_expected   = {0, comm.size_signed() - 1, true, comm.size_signed()};
    Aggregate              agg2_expected   = {42, comm.size_signed() - 1 + 42, false, comm.size_signed()};
    std::vector<Aggregate> expected_result = {agg1_expected, agg2_expected};

    auto result = comm.allreduce(send_buf(input), op(my_op, kamping::commutative)).extract_recv_buffer();

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result, expected_result);
}
