
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

#include "../test_assertions.hpp"

#include "gmock/gmock.h"

#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/scan.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(ScanTest, scan_single) {
    Communicator comm;

    int input = 42;

    auto result = comm.scan_single(send_buf(input), op(kamping::ops::plus<>{}));
    static_assert(std::is_same_v<decltype(result), decltype(input)>);
    int expected_result = (comm.rank_signed() + 1) * 42;
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_single_vector_of_size_1) {
    Communicator comm;

    std::vector<int> input = {42};

    auto result = comm.scan_single(send_buf(input), op(kamping::ops::plus<>{}));
    static_assert(std::is_same_v<decltype(result), decltype(input)::value_type>);
    int expected_result = (comm.rank_signed() + 1) * 42;
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_single_vector_of_size_2) {
    Communicator comm;

    std::vector<int> input = {42, 1};

#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    EXPECT_KASSERT_FAILS(
        (comm.scan_single(send_buf(input), op(kamping::ops::plus<>{}))),
        "The send buffer has to be of size 1 on all ranks."
    );
#endif
}

TEST(ScanTest, scan_explicit_send_recv_count_smaller_than_send_buffer_size) {
    Communicator comm;

    std::vector<int> input = {42, 1, 1, 1, 1};

    auto result   = comm.scan(send_buf(input), send_recv_count(2), op(kamping::ops::plus<>{}));
    auto recv_buf = result.extract_recv_buffer();
    EXPECT_EQ(recv_buf.size(), 2);
    EXPECT_THAT(recv_buf, ElementsAre((comm.rank_signed() + 1) * 42, (comm.rank_signed() + 1)));
}

TEST(ScanTest, scan_explicit_send_recv_count_out_value_not_taken_into_account) {
    Communicator comm;

    std::vector<int> input           = {42, 1};
    int              send_recv_count = -1;

    auto result   = comm.scan(send_buf(input), send_recv_count_out(send_recv_count), op(kamping::ops::plus<>{}));
    auto recv_buf = result.extract_recv_buffer();
    EXPECT_EQ(recv_buf.size(), 2);
    EXPECT_EQ(send_recv_count, 2);
    EXPECT_THAT(recv_buf, ElementsAre((comm.rank_signed() + 1) * 42, (comm.rank_signed() + 1)));
}

TEST(ScanTest, scan_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input = {42, 1};

    auto result   = comm.scan(send_buf(input), send_recv_count(2), op(kamping::ops::plus<>{}));
    auto recv_buf = result.extract_recv_buffer();
    EXPECT_THAT(recv_buf, ElementsAre((comm.rank_signed() + 1) * 42, (comm.rank_signed() + 1)));
}

TEST(ScanTest, scan_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto result = comm.scan(send_buf(input), op(kamping::ops::plus<>{})).extract_recv_buffer();
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {
        ((comm.rank_signed() + 1) * comm.rank_signed()) / 2,
        (comm.rank_signed() + 1) * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.scan(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result));
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {
        ((comm.rank_signed() + 1) * comm.rank_signed()) / 2,
        (comm.rank_signed() + 1) * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_builtin_op_on_non_builtin_type) {
    Communicator comm;

    struct MyInt {
        MyInt() noexcept : _value(0) {}
        MyInt(int value) noexcept : _value(value) {}
        int _value;
        int operator+(MyInt const& rhs) const noexcept {
            return this->_value + rhs._value;
        }
        bool operator==(MyInt const& rhs) const noexcept {
            return this->_value == rhs._value;
        }
    };
    std::vector<MyInt> input = {comm.rank_signed(), 42};

    auto result =
        comm.scan(send_buf(input), op(kamping::ops::plus<>{}, kamping::ops::commutative)).extract_recv_buffer();
    EXPECT_EQ(result.size(), 2);
    std::vector<MyInt> expected_result = {
        ((comm.rank_signed() + 1) * comm.rank_signed()) / 2,
        (comm.rank_signed() + 1) * 42};
    EXPECT_EQ(result, expected_result);
}

int add_plus_42_function(int const& lhs, int const& rhs) {
    return lhs + rhs + 42;
}

TEST(ScanTest, scan_custom_operation_on_builtin_type) {
    Communicator comm;

    auto add_plus_42_lambda = [](auto const& lhs, auto const& rhs) {
        return lhs + rhs + 42;
    };

    std::vector<int> input = {0, 17, 8};

    { // use function ptr
        auto result =
            comm.scan(send_buf(input), op(add_plus_42_function, kamping::ops::commutative)).extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            (comm.rank_signed() + 1) * 0 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 17 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 8 + (comm.rank_signed()) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use lambda
        auto result =
            comm.scan(send_buf(input), op(add_plus_42_lambda, kamping::ops::commutative)).extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            (comm.rank_signed() + 1) * 0 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 17 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 8 + (comm.rank_signed()) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use lambda inline
        auto result =
            comm.scan(
                    send_buf(input),
                    op([](auto const& lhs, auto const& rhs) { return lhs + rhs + 42; }, kamping::ops::commutative)
            )
                .extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            (comm.rank_signed() + 1) * 0 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 17 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 8 + (comm.rank_signed()) * 42};
        EXPECT_EQ(result, expected_result);
    }

    { // use function object
        struct MySum42 {
            int operator()(int const& lhs, int const& rhs) {
                return lhs + rhs + 42;
            }
        };
        auto result = comm.scan(send_buf(input), op(MySum42{}, kamping::ops::commutative)).extract_recv_buffer();

        EXPECT_EQ(result.size(), 3);
        std::vector<int> expected_result = {
            (comm.rank_signed() + 1) * 0 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 17 + (comm.rank_signed()) * 42,
            (comm.rank_signed() + 1) * 8 + (comm.rank_signed()) * 42};
        EXPECT_EQ(result, expected_result);
    }
}

TEST(ScanTest, scan_custom_operation_on_builtin_type_non_commutative) {
    Communicator comm;

    auto get_right = [](auto const&, auto const& rhs) {
        return rhs;
    };

    std::vector<int> input = {comm.rank_signed() + 17};

    auto result = comm.scan(send_buf(input), op(get_right, kamping::ops::non_commutative)).extract_recv_buffer();

    EXPECT_EQ(result.size(), 1);
    std::vector<int> expected_result = {comm.rank_signed() + 17};
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_custom_operation_on_custom_type) {
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

    Aggregate              agg1_expected   = {0, comm.rank_signed(), true, comm.rank_signed() + 1};
    Aggregate              agg2_expected   = {42, comm.rank_signed() + 42, false, comm.rank_signed() + 1};
    std::vector<Aggregate> expected_result = {agg1_expected, agg2_expected};

    auto result = comm.scan(send_buf(input), op(my_op, kamping::ops::commutative)).extract_recv_buffer();

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result, expected_result);
}

TEST(ScanTest, scan_default_container_type) {
    Communicator<OwnContainer> comm;
    std::vector<int>           input = {comm.rank_signed(), 42};

    // This just has to compile
    OwnContainer<int> result = comm.scan(send_buf(input), op(kamping::ops::plus<>{})).extract_recv_buffer();
}

TEST(ScanTest, single_element_with_given_recv_buf_bigger_than_required) {
    Communicator     comm;
    std::vector<int> input               = {1};
    int              expected_recv_value = comm.rank_signed() + 1;

    {
        // recv buffer will be resized as policy is resize_to_fit
        std::vector<int> recv_buffer(2, -1);
        comm.scan(send_buf(input), recv_buf<resize_to_fit>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_EQ(recv_buffer.front(), expected_recv_value);
    }
    {
        // recv buffer will not be resized as it is large enough and policy is grow_only
        std::vector<int> recv_buffer(2, -1);
        comm.scan(send_buf(input), recv_buf<grow_only>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_THAT(recv_buffer, ElementsAre(expected_recv_value, -1));
    }
    {
        // recv buffer will not be resized as the policy is no_resize
        std::vector<int> recv_buffer(2, -1);
        comm.scan(send_buf(input), recv_buf<no_resize>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_THAT(recv_buffer, ElementsAre(expected_recv_value, -1));
    }
    {
        // recv buffer will not be resized as the policy is no_resize (default)
        std::vector<int> recv_buffer(2, -1);
        comm.scan(send_buf(input), recv_buf(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_THAT(recv_buffer, ElementsAre(expected_recv_value, -1));
    }
}

TEST(ScanTest, single_element_with_given_recv_buf_smaller_than_required) {
    Communicator     comm;
    std::vector<int> input = {1};
    std::vector<int> expected_recv_buffer{comm.rank_signed() + 1};

    {
        // recv buffer will be resized as policy is resize_to_fit
        std::vector<int> recv_buffer;
        comm.scan(send_buf(input), recv_buf<resize_to_fit>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
    {
        // recv buffer will be resized as policy is grow_only and buffer is too small
        std::vector<int> recv_buffer;
        comm.scan(send_buf(input), recv_buf<grow_only>(recv_buffer), op(kamping::ops::plus<>{}));
        EXPECT_EQ(recv_buffer, expected_recv_buffer);
    }
#if KASSERT_ENABLED(KAMPING_ASSERTION_LEVEL_NORMAL)
    {
        // recv buffer will not be resized as the policy is no_resize
        std::vector<int> recv_buffer;
        EXPECT_KASSERT_FAILS(
            comm.scan(send_buf(input), recv_buf<no_resize>(recv_buffer), op(kamping::ops::plus<>{})),
            ""
        );
    }
    {
        // recv buffer will not be resized as the policy is no_resize (default)
        std::vector<int> recv_buffer;
        EXPECT_KASSERT_FAILS(
            comm.scan(send_buf(input), recv_buf<no_resize>(recv_buffer), op(kamping::ops::plus<>{})),
            ""
        );
    }
#endif
}
