#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/v2/collectives/scan.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views/resize_view.hpp"

using namespace ::kamping;

class ScanTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    int rank_;
    int size_;
};

TEST_F(ScanTest, scan_sum) {
    // Inclusive prefix sum: rank r receives sum of ranks 0..r
    std::vector<int> send_data = {rank_ + 1};
    std::vector<int> recv_data(1);

    auto [s, r] = kamping::v2::scan(send_data, recv_data, MPI_SUM);

    // Inclusive prefix sum of (0+1, 1+1, ..., rank+1)
    int expected = (rank_ + 1) * (rank_ + 2) / 2;
    EXPECT_EQ(r[0], expected);
}

TEST_F(ScanTest, scan_multi_element) {
    std::vector<int> send_data = {rank_, rank_ + 1};
    std::vector<int> recv_data(2);

    auto [s, r] = kamping::v2::scan(send_data, recv_data, MPI_SUM);

    int expected_0 = rank_ * (rank_ + 1) / 2;
    int expected_1 = (rank_ + 1) * (rank_ + 2) / 2;
    EXPECT_EQ(r[0], expected_0);
    EXPECT_EQ(r[1], expected_1);
}

TEST_F(ScanTest, scan_default_op) {
    std::vector<int> send_data = {rank_ + 1};
    std::vector<int> recv_data(1);

    auto [s, r] = kamping::v2::scan(send_data, recv_data);

    int expected = (rank_ + 1) * (rank_ + 2) / 2;
    EXPECT_EQ(r[0], expected);
}

TEST_F(ScanTest, scan_resize) {
    std::vector<int> send_data = {rank_ + 1, rank_ + 2};
    std::vector<int> recv_data;

    auto [s, r] = kamping::v2::scan(send_data, recv_data | kamping::v2::views::resize, MPI_SUM);

    ASSERT_EQ(static_cast<int>(r.size()), 2);
    int expected_0 = (rank_ + 1) * (rank_ + 2) / 2;
    int expected_1 = (rank_ + 1) * (rank_ + 2) / 2 + rank_ + 1;
    EXPECT_EQ(r[0], expected_0);
    EXPECT_EQ(r[1], expected_1);
}

TEST_F(ScanTest, scan_inplace) {
    std::vector<int> data = {rank_ + 1};

    kamping::v2::scan(kamping::v2::inplace, data, MPI_SUM);

    int expected = (rank_ + 1) * (rank_ + 2) / 2;
    EXPECT_EQ(data[0], expected);
}

TEST_F(ScanTest, scan_max) {
    std::vector<int> send_data = {rank_};
    std::vector<int> recv_data(1);

    auto [s, r] = kamping::v2::scan(send_data, recv_data, MPI_MAX);

    // Inclusive prefix max: result on rank r is max(0..r) = r
    EXPECT_EQ(r[0], rank_);
}
