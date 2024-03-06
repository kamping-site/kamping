#include <unordered_map>

#include <gtest/gtest.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/utils/flatten.hpp>

using namespace kamping;

TEST(FlattenTest, basic) {
    Communicator                              comm;
    std::unordered_map<int, std::vector<int>> sparse_send_buf;
    for (int dst = 0; dst < comm.size_signed(); dst++) {
        sparse_send_buf.emplace(dst, std::vector<int>(comm.rank() + 1, comm.rank_signed()));
    }

    auto [recv_buf, recv_displs] = with_flattened(sparse_send_buf, comm).do_([&](auto... flattened) {
      return comm.alltoallv(std::move(flattened)..., recv_counts_out());
    });
}
