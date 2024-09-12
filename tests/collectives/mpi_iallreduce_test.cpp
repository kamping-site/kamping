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

//TEST(IallreduceTest, allreduce_no_receive_buffer) {
//    Communicator comm;
//
//    std::vector<int> input = {comm.rank_signed(), 42};
//
//    auto non_blocking_result = comm.iallreduce(send_buf(input), op(kamping::ops::plus<>{}));
//    auto result = non_blocking_result.wait();
//
//    EXPECT_EQ(*non_blocking_result.get_request_ptr(), MPI_REQUEST_NULL);
//    std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
//    EXPECT_EQ(result, expected_result);
//}

TEST(AllreduceTest, allreduce_with_receive_buffer) {
    Communicator comm;

    std::vector<int> input = {comm.rank_signed(), 42};
    std::vector<int> result;

    comm.iallreduce(send_buf(input), op(kamping::ops::plus<>{}), recv_buf<resize_to_fit>(result));
    //non_blocking_result.wait();


    //EXPECT_EQ(result.size(), 2);

    //std::vector<int> expected_result = {(comm.size_signed() * (comm.size_signed() - 1)) / 2, comm.size_signed() * 42};
    //EXPECT_EQ(result, expected_result);
}
