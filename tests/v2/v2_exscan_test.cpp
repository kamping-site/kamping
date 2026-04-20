#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/exscan.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/resize_view.hpp"

using namespace ::kamping;

class ExscanTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    int rank_;
    int size_;
};

TEST_F(ExscanTest, exscan_sum) {
    // Exclusive prefix sum: rank r receives sum of ranks 0..r-1.
    // Rank 0 receive buffer is undefined — skip the check for rank 0.
    std::vector<int> send_data = {rank_ + 1};
    std::vector<int> recv_data(1, 0);

    auto [s, r] = kamping::v2::exscan(send_data, recv_data, MPI_SUM);

    if (rank_ > 0) {
        int expected = rank_ * (rank_ + 1) / 2;
        EXPECT_EQ(r[0], expected);
    }
}

TEST_F(ExscanTest, exscan_multi_element) {
    std::vector<int> send_data = {rank_, rank_ + 1};
    std::vector<int> recv_data(2, 0);

    auto [s, r] = kamping::v2::exscan(send_data, recv_data, MPI_SUM);

    if (rank_ > 0) {
        int expected_0 = rank_ * (rank_ - 1) / 2;
        int expected_1 = rank_ * (rank_ + 1) / 2;
        EXPECT_EQ(r[0], expected_0);
        EXPECT_EQ(r[1], expected_1);
    }
}

TEST_F(ExscanTest, exscan_default_op) {
    std::vector<int> send_data = {rank_ + 1};
    std::vector<int> recv_data(1, 0);

    auto [s, r] = kamping::v2::exscan(send_data, recv_data);

    if (rank_ > 0) {
        int expected = rank_ * (rank_ + 1) / 2;
        EXPECT_EQ(r[0], expected);
    }
}

TEST_F(ExscanTest, exscan_resize) {
    std::vector<int> send_data = {rank_ + 1, rank_ + 2};
    std::vector<int> recv_data;

    auto [s, r] = kamping::v2::exscan(send_data, recv_data | kamping::v2::views::resize, MPI_SUM);

    ASSERT_EQ(static_cast<int>(r.size()), 2);
    if (rank_ > 0) {
        int expected_0 = rank_ * (rank_ + 1) / 2;
        int expected_1 = rank_ * (rank_ + 1) / 2 + rank_;
        EXPECT_EQ(r[0], expected_0);
        EXPECT_EQ(r[1], expected_1);
    }
}

TEST_F(ExscanTest, exscan_inplace) {
    std::vector<int> data = {rank_ + 1};

    kamping::v2::exscan(kamping::v2::inplace, data, MPI_SUM);

    if (rank_ > 0) {
        int expected = rank_ * (rank_ + 1) / 2;
        EXPECT_EQ(data[0], expected);
    }
}
