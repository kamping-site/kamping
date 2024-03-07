#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/utils/flatten.hpp>

#include "../helpers_for_testing.hpp"

using namespace kamping;

TEST(FlattenTest, basic) {
    Communicator                              comm;
    std::unordered_map<int, std::vector<int>> sparse_send_buf;
    for (int dst = 0; dst < comm.size_signed(); dst++) {
        sparse_send_buf.emplace(dst, std::vector<int>(1, dst));
    }

    auto [recv_buf, recv_counts, recv_displs] = with_flattened(sparse_send_buf, comm).do_([&](auto... flattened) {
        return comm.alltoallv(std::move(flattened)..., recv_counts_out(), recv_displs_out());
    });

    EXPECT_EQ(recv_buf.size(), comm.size());
    EXPECT_THAT(recv_buf, ::testing::Each(comm.rank_signed()));
    EXPECT_THAT(recv_counts.size(), comm.size());
    EXPECT_THAT(recv_counts, ::testing::Each(1));
    EXPECT_EQ(recv_displs, testing::iota_container_n(comm.size(), 0));
}
