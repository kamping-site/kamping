// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../helpers_for_testing.hpp"
#include "kamping/collectives/iallreduce.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(IallreduceTest, iallreduce_no_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto non_blocking_result = comm.iallreduce(send_buf(input), op(kamping::ops::plus<>{}));
    auto result              = non_blocking_result.wait();

    EXPECT_EQ(*non_blocking_result.get_request_ptr(), MPI_REQUEST_NULL);
    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(IallreduceTest, iallreduce_no_receive_buffer_with_test) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};

    auto non_blocking_result = comm.iallreduce(send_buf(input), op(kamping::ops::plus<>{}));

    auto result = non_blocking_result.test();
    while (!result.has_value()) {
        EXPECT_NE(*non_blocking_result.get_request_ptr(), MPI_REQUEST_NULL);
        result = non_blocking_result.test();
    }

    EXPECT_EQ(*non_blocking_result.get_request_ptr(), MPI_REQUEST_NULL);
    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(IallreduceTest, iallreduce_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.iallreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result)).wait();

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(IallreduceTest, iallreduce_with_receive_buffer_resize_too_big) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result(10, -1);

    comm.iallreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result)).wait();
    EXPECT_EQ(result.size(), 2);

    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    EXPECT_EQ(result, expected_result);
}

TEST(IallreduceTest, iallreduce_with_receive_buffer_no_resize_and_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input  = {1, 2, 3, 4};
    std::vector<int> result = {42, 42};

    comm.iallreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<no_resize>(result), send_recv_count(1))
        .wait();
    EXPECT_THAT(result, ElementsAre(comm.size(), 42));
}

TEST(IallreduceTest, iallreduce_with_receive_buffer_grow_only_and_explicit_send_recv_count) {
    Communicator comm;

    std::vector<int> input  = {1, 2, 3, 4};
    std::vector<int> result = {42, 42};

    comm.iallreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<grow_only>(result), send_recv_count(1))
        .wait();
    EXPECT_THAT(result, ElementsAre(comm.size(), 42));
}

TEST(IallreduceTest, iallreduce_move_send_buf_to_call) {
    Communicator comm;

    std::vector<int> input = {1, 2, 3, 4};
    std::vector<int> expected_recv_buf{
        comm.size_signed() * 1,
        comm.size_signed() * 2,
        comm.size_signed() * 3,
        comm.size_signed() * 4};
    auto const expected_send_buf = input;

    auto [recv_buf, send_buf] = comm.iallreduce(send_buf_out(std::move(input)), op(kamping::ops::plus<>{})).wait();
    EXPECT_EQ(send_buf, expected_send_buf);
    EXPECT_EQ(recv_buf, expected_recv_buf);
}

TEST(IallreduceTest, iallreduce_move_send_buf_and_recv_buf_to_call) {
    Communicator comm;

    std::vector<int> input = {1, 2, 3, 4};
    std::vector<int> output(6, 42);
    std::vector<int> expected_recv_buf{
        comm.size_signed() * 1,
        comm.size_signed() * 2,
        comm.size_signed() * 3,
        comm.size_signed() * 4,
        42,
        42};
    auto const expected_send_buf = input;

    auto nonblocking_result =
        comm.iallreduce(recv_buf_out(std::move(output)), send_buf_out(std::move(input)), op(kamping::ops::plus<>{}));
    // clear to rule out improper useage of move semantics
    input.clear();
    output.clear();
    auto result = nonblocking_result.wait();
    input       = result.extract_send_buf();
    output      = result.extract_recv_buf();

    EXPECT_EQ(input, expected_send_buf);
    EXPECT_EQ(output, expected_recv_buf);
}
