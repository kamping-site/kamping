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

#include <cstddef>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "kamping/plugin/alltoall_dispatch.hpp"

using namespace ::kamping;
using namespace ::testing;
using namespace ::plugin;
using namespace ::dispatch_alltoall;

using namespace grid_plugin_helpers;

TEST(DispatchAlltoallTest, alltoallv_dispatch_single_element) {
    Communicator<std::vector, plugin::GridCommunicator, plugin::DispatchAlltoall> comm;

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto result = comm.alltoallv_dispatch(send_buf(input), kamping::send_counts(send_counts));
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(comm.rank_signed()));
}

TEST(DispatchAlltoallTest, alltoallv_disptach_single_element_minimum_threshold) {
    Communicator<std::vector, plugin::GridCommunicator, plugin::DispatchAlltoall> comm;

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto [result, recv_counts] = comm.alltoallv_dispatch(
        send_buf(input),
        kamping::send_counts(send_counts),
        comm_volume_threshold(0),
        recv_counts_out()
    );
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(comm.rank_signed()));
    EXPECT_EQ(recv_counts.size(), comm.size());
    EXPECT_THAT(recv_counts, Each(1));
}

TEST(DispatchAlltoallTest, alltoallv_disptach_single_element_maximum_threshold) {
    Communicator<std::vector, plugin::GridCommunicator, plugin::DispatchAlltoall> comm;

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto [result, recv_counts] = comm.alltoallv_dispatch(
        send_buf(input),
        kamping::send_counts(send_counts),
        comm_volume_threshold(std::numeric_limits<size_t>::max()),
        recv_counts_out()
    );
    EXPECT_EQ(result.size(), comm.size());
    EXPECT_THAT(result, Each(comm.rank_signed()));
    EXPECT_EQ(recv_counts.size(), comm.size());
    EXPECT_THAT(recv_counts, Each(1));
}

TEST(DispatchAlltoallGridTest, alltoallv_single_element_st_binding) {
    Communicator<std::vector, plugin::GridCommunicator> comm;
    auto                                                grid_comm = comm.make_grid_communicator();

    std::vector<int> input(comm.size());
    std::iota(input.begin(), input.end(), 0);
    std::vector<int> send_counts(comm.size(), 1);

    auto [recv_buf, recv_counts] =
        grid_comm.alltoallv(recv_counts_out(), send_buf(input), kamping::send_counts(send_counts));

    EXPECT_EQ(recv_buf.size(), comm.size());
    EXPECT_THAT(recv_buf, Each(comm.rank_signed()));
    EXPECT_EQ(recv_counts.size(), comm.size());
    EXPECT_THAT(recv_counts, Each(1));
}
