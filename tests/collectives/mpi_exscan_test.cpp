
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
#include "kamping/collectives/exscan.hpp"
#include "kamping/communicator.hpp"
#include "kamping/parameter_factories.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(exscanTest, exscan_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto result = comm.exscan(send_buf(input), op(kamping::ops::plus<>{})).extract_recv_buffer();
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(exscanTest, exscan_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf(result));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(exscanTest, exscan_builtin_op_on_non_builtin_type) {
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
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}, kamping::commutative), values_on_rank_0(MyInt{0}))
            .extract_recv_buffer();
    EXPECT_EQ(result.size(), 2);
    std::vector<MyInt> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(exscanTest, exscan_no_identity_values_on_rank_0) {
    Communicator comm;

    std::vector<int> input = {0};

    auto result =
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), values_on_rank_0(1337)).extract_recv_buffer();
    EXPECT_EQ(result.size(), 1);

    std::vector<int> expected_result;
    if (comm.rank() == 0) {
        expected_result.push_back(1337);
    } else {
        expected_result.push_back(0);
    }
    EXPECT_EQ(result, expected_result);
}

int add_plus_42_function(int const& lhs, int const& rhs) {
    return lhs + rhs + 42;
}

TEST(exscanTest, exscan_custom_operation_on_builtin_type) {
    Communicator comm;

    auto add_plus_42_lambda = [](auto const& lhs, auto const& rhs) {
        return lhs + rhs + 42;
    };

    std::vector<int> input = {0, 17, 8};

    { // use function ptr
        auto result =
            comm.exscan(send_buf(input), op(add_plus_42_function, kamping::commutative), values_on_rank_0({0, 1, 2}))
                .extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        if (comm.rank() == 0) {
            EXPECT_EQ(result, (std::vector<int>{0, 1, 2}));
        } else {
            std::vector<int> expected_result = {
                comm.rank_signed() * 0 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 17 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 8 + (comm.rank_signed() - 1) * 42};
            EXPECT_EQ(result, expected_result);
        }
    }

    { // use lambda
        auto result =
            comm.exscan(send_buf(input), op(add_plus_42_lambda, kamping::commutative), values_on_rank_0({0, 1, 2}))
                .extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        if (comm.rank() == 0) {
            EXPECT_EQ(result, (std::vector<int>{0, 1, 2}));
        } else {
            std::vector<int> expected_result = {
                comm.rank_signed() * 0 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 17 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 8 + (comm.rank_signed() - 1) * 42};
            EXPECT_EQ(result, expected_result);
        }
    }

    { // use lambda inline
        auto result = comm.exscan(
                              send_buf(input),
                              op([](auto const& lhs, auto const& rhs) { return lhs + rhs + 42; }, kamping::commutative),
                              values_on_rank_0({0, 1, 2})
        )
                          .extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        if (comm.rank() == 0) {
            EXPECT_EQ(result, (std::vector<int>{0, 1, 2}));
        } else {
            std::vector<int> expected_result = {
                comm.rank_signed() * 0 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 17 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 8 + (comm.rank_signed() - 1) * 42};
            EXPECT_EQ(result, expected_result);
        }
    }

    { // use function object
        struct MySum42 {
            int operator()(int const& lhs, int const& rhs) {
                return lhs + rhs + 42;
            }
        };
        auto result = comm.exscan(send_buf(input), op(MySum42{}, kamping::commutative), values_on_rank_0({0, 1, 2}))
                          .extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        if (comm.rank() == 0) {
            EXPECT_EQ(result, (std::vector<int>{0, 1, 2}));
        } else {
            std::vector<int> expected_result = {
                comm.rank_signed() * 0 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 17 + (comm.rank_signed() - 1) * 42,
                comm.rank_signed() * 8 + (comm.rank_signed() - 1) * 42};
            EXPECT_EQ(result, expected_result);
        }
    }
}

TEST(exscanTest, exscan_custom_operation_on_builtin_type_non_commutative) {
    Communicator comm;

    auto get_right = [](auto const&, auto const& rhs) {
        return rhs;
    };

    std::vector<int> input = {comm.rank_signed() + 17};

    auto result = comm.exscan(send_buf(input), op(get_right, kamping::non_commutative), values_on_rank_0(0))
                      .extract_recv_buffer();

    EXPECT_EQ(result.size(), 1);
    if (comm.rank() == 0) {
        EXPECT_EQ(result, (std::vector<int>{0}));
    } else {
        std::vector<int> expected_result = {comm.rank_signed() - 1 + 17};
        EXPECT_EQ(result, expected_result);
    }
}

/// @todo Once our helper macros support checking for KASSERTs which are thrown on some ranks only, write a test for
/// and values_on_rank_0 size which is not 1 and not equal the length of the recv_buf.
