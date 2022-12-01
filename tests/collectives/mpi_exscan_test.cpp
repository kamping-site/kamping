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
#include "kamping/named_parameters.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(ExscanTest, exscan_single_without_recv_buf) {
    Communicator comm;

    int input = 42;

    auto result = comm.exscan_single(send_buf(input), op(kamping::ops::plus<>{}));
    static_assert(std::is_same_v<decltype(result), decltype(input)>);
    if (comm.rank() != 0) {
        int expected_result = comm.rank_signed() * 42;
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, exscan_single_with_recv_buf) {
    Communicator comm;

    int input = 42;
    int result;

    comm.exscan_single(send_buf(input), recv_buf(result), op(kamping::ops::plus<>{}));
    if (comm.rank() != 0) {
        int expected_result = comm.rank_signed() * 42;
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, exscan_single_values_on_rank_0) {
    Communicator comm;

    int input = 42;
    int result;

    comm.exscan_single(send_buf(input), recv_buf(result), op(kamping::ops::plus<>{}), values_on_rank_0(0));
    int expected_result = comm.rank_signed() * 42;
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, exscan_single_vector_of_size_1_with_recv_buf) {
    Communicator comm;

    std::vector<int> input = {42};
    int              result;

    comm.exscan_single(send_buf(input), recv_buf(result), op(kamping::ops::plus<>{}));
    if (comm.rank() != 0) {
        int expected_result = comm.rank_signed() * 42;
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, exscan_single_vector_of_size_1_without_recv_buf) {
    Communicator comm;

    std::vector<int> input = {42};

    auto result = comm.exscan_single(send_buf(input), op(kamping::ops::plus<>{}));
    static_assert(std::is_same_v<decltype(result), decltype(input)::value_type>);
    if (comm.rank() != 0) {
        int expected_result = comm.rank_signed() * 42;
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, exscan_single_vector_of_size_2_with_recv_buf) {
    Communicator comm;

    {
        std::vector<int>     input = {42, 1};
        [[maybe_unused]] int result;

        EXPECT_KASSERT_FAILS(
            (comm.exscan_single(send_buf(input), recv_buf(result), op(kamping::ops::plus<>{}))),
            "The send buffer has to be of size 1 on all ranks."
        );
    }

    {
        std::vector<int>                  input = {42, 1};
        [[maybe_unused]] std::vector<int> result;

        EXPECT_KASSERT_FAILS(
            (comm.exscan_single(send_buf(input), recv_buf(result), op(kamping::ops::plus<>{}))),
            "The send buffer has to be of size 1 on all ranks."
        );
    }
}

TEST(ExscanTest, exscan_single_vector_of_size_2_without_recv_buf) {
    Communicator comm;

    std::vector<int> input = {42, 1};

    EXPECT_KASSERT_FAILS(
        (comm.exscan_single(send_buf(input), op(kamping::ops::plus<>{}))),
        "The send buffer has to be of size 1 on all ranks."
    );
}

TEST(ExscanTest, no_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto result = comm.exscan(send_buf(input), op(kamping::ops::plus<>{})).extract_recv_buffer();
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.exscan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf(result));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, builtin_op_on_non_builtin_type) {
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
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}, kamping::ops::commutative), values_on_rank_0(MyInt{0}))
            .extract_recv_buffer();
    EXPECT_EQ(result.size(), 2);
    std::vector<MyInt> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ExscanTest, identity_not_auto_deducible_and_no_values_on_rank_0_provided) {
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
        comm.exscan(send_buf(input), op(kamping::ops::plus<>{}, kamping::ops::commutative)).extract_recv_buffer();
    EXPECT_EQ(result.size(), 2);
    std::vector<MyInt> expected_result = {((comm.rank_signed() - 1) * comm.rank_signed()) / 2, comm.rank_signed() * 42};
    if (comm.rank() != 0) { // The result of this exscan() is not defined on rank 0.
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, non_identity_values_on_rank_0) {
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

TEST(ExscanTest, custom_operation_on_builtin_type) {
    Communicator comm;

    auto add_plus_42_lambda = [](auto const& lhs, auto const& rhs) {
        return lhs + rhs + 42;
    };

    std::vector<int> input = {0, 17, 8};

    { // use function ptr
        auto result = comm.exscan(
                              send_buf(input),
                              op(add_plus_42_function, kamping::ops::commutative),
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

    { // use lambda
        auto result =
            comm.exscan(send_buf(input), op(add_plus_42_lambda, kamping::ops::commutative), values_on_rank_0({0, 1, 2}))
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
        auto result =
            comm.exscan(
                    send_buf(input),
                    op([](auto const& lhs, auto const& rhs) { return lhs + rhs + 42; }, kamping::ops::commutative),
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
        auto result =
            comm.exscan(send_buf(input), op(MySum42{}, kamping::ops::commutative), values_on_rank_0({0, 1, 2}))
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

TEST(ExscanTest, custom_operation_on_builtin_type_non_commutative) {
    Communicator comm;

    auto get_right = [](auto const&, auto const& rhs) {
        return rhs;
    };

    std::vector<int> input = {comm.rank_signed() + 17};

    auto result = comm.exscan(send_buf(input), op(get_right, kamping::ops::non_commutative), values_on_rank_0(0))
                      .extract_recv_buffer();

    EXPECT_EQ(result.size(), 1);
    if (comm.rank() == 0) {
        EXPECT_EQ(result, (std::vector<int>{0}));
    } else {
        std::vector<int> expected_result = {comm.rank_signed() - 1 + 17};
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ExscanTest, default_container_type) {
    Communicator<OwnContainer> comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    // This just has to compile
    OwnContainer<int> result = comm.exscan(send_buf(input), op(kamping::ops::plus<>{})).extract_recv_buffer();
}

/// @todo Once our helper macros support checking for KASSERTs which are thrown on some ranks only, write a test for
/// and values_on_rank_0 size which is not 1 and not equal to the length of recv_buf.
