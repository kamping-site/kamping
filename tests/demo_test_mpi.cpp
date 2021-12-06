#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <numeric>

struct ExampleTest: ::testing::Test {
    void SetUp() override {
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    int rank;
    int size;
};

TEST_F(ExampleTest, SingleElementGatherWorks) {
    std::vector<int> recv_buf;
    if (rank == 0) {
        recv_buf.resize(size);
    }
    MPI_Gather(&rank, 1, MPI_INT, recv_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::vector<int> expected(size);
        std::iota(expected.begin(), expected.end(), 0);
        EXPECT_THAT(recv_buf, testing::Eq(expected));
    }
}
