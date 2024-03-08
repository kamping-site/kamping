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
//
#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/utils/flatten.hpp>

#include "../helpers_for_testing.hpp"

using namespace kamping;

TEST(FlattenTest, basic_unorderd_map) {
    Communicator                              comm;
    std::unordered_map<int, std::vector<int>> sparse_send_buf;
    for (int dst = 0; dst < comm.size_signed(); dst++) {
        sparse_send_buf.emplace(dst, std::vector<int>(1, dst));
    }

    auto [recv_buf, recv_counts, recv_displs] =
        with_flattened(sparse_send_buf, comm.size()).call([&](auto... flattened) {
            return comm.alltoallv(std::move(flattened)..., recv_counts_out(), recv_displs_out());
        });

    EXPECT_EQ(recv_buf.size(), comm.size());
    EXPECT_THAT(recv_buf, ::testing::Each(comm.rank_signed()));
    EXPECT_THAT(recv_counts.size(), comm.size());
    EXPECT_THAT(recv_counts, ::testing::Each(1));
    EXPECT_EQ(recv_displs, testing::iota_container_n(comm.size(), 0));
}

TEST(FlattenTest, basic_vector_of_vectors) {
    Communicator                  comm;
    std::vector<std::vector<int>> nested_send_buf(comm.size());
    for (int i = 0; i < comm.size_signed(); i++) {
        nested_send_buf[asserting_cast<size_t>(i)] = std::vector<int>(1, i);
    }

    auto [recv_buf, recv_counts, recv_displs] = with_flattened(nested_send_buf).call([&](auto... flattened) {
        return comm.alltoallv(std::move(flattened)..., recv_counts_out(), recv_displs_out());
    });

    EXPECT_EQ(recv_buf.size(), comm.size());
    EXPECT_THAT(recv_buf, ::testing::Each(comm.rank_signed()));
    EXPECT_THAT(recv_counts.size(), comm.size());
    EXPECT_THAT(recv_counts, ::testing::Each(1));
    EXPECT_EQ(recv_displs, testing::iota_container_n(comm.size(), 0));
}
