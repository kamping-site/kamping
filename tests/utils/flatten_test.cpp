#include <unordered_map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/communicator.hpp>
#include <kamping/utils/flatten.hpp>

using namespace kamping;

TEST(FlattenTest, basic) {
    Communicator                              comm;
    std::unordered_map<int, std::vector<int>> sparse_send_buf;
    for (int dst = 0; dst < comm.size_signed(); dst++) {
        sparse_send_buf.emplace(dst, std::vector<int>(1, dst));
    }

    auto [recv_buf, recv_displs] = with_flattened(sparse_send_buf, comm).do_([&](auto... flattened) {
        return comm.alltoallv(std::move(flattened)..., recv_counts_out());
    });
    // print recv_buf
    for (auto const& elem: recv_buf) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    kamping::Span recv_buf_view(recv_buf);
    for (size_t i = 0; i < comm.size(); i++) {
        size_t chunk_size;
        if (i == comm.size() - 1) {
            chunk_size = recv_buf_view.size() - asserting_cast<size_t>(recv_displs[i]);
        } else {
            chunk_size = asserting_cast<size_t>(recv_displs[i + 1] - recv_displs[i]);
        }
        auto msg_from_rank_i = recv_buf_view.subspan(static_cast<size_t>(recv_displs[i]), chunk_size);
        EXPECT_EQ(msg_from_rank_i.size(), 1);
        EXPECT_THAT(msg_from_rank_i, ::testing::Each(comm.rank_signed()));
    }
}
