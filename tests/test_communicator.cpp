#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

struct CommunicatorTest : Test {
    void SetUp() override {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    int rank;
    int size;
};

TEST_F(CommunicatorTest, EmptyConstructor) {
    Communicator comm;

    EXPECT_EQ(comm.mpi_communicator(), MPI_COMM_WORLD);
    EXPECT_EQ(comm.rank(), rank);
    EXPECT_EQ(comm.size(), size);
}

TEST_F(CommunicatorTest, ConstructorWithMPICommunicator) {
    Communicator comm(MPI_COMM_SELF);

    int self_rank;
    int self_size;

    MPI_Comm_size(MPI_COMM_SELF, &self_size);
    MPI_Comm_rank(MPI_COMM_SELF, &self_rank);

    EXPECT_EQ(comm.mpi_communicator(), MPI_COMM_SELF);
    EXPECT_EQ(comm.rank(), self_rank);
    EXPECT_EQ(comm.size(), self_size);
}

TEST_F(CommunicatorTest, ConstructorWithMPICommunicatorAndRoot) {
    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i < 0 || i >= size) {
            EXPECT_DEATH(Communicator comm(MPI_COMM_WORLD, i), ".*");
        } else {
            Communicator comm(MPI_COMM_WORLD, i);
            ASSERT_EQ(comm.root(), i);
        }
    }
}

TEST_F(CommunicatorTest, RankAdvanceBoundCheck) {
    Communicator comm;

    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i + rank < 0 || i + rank >= size) {
            EXPECT_DEATH(comm.compute_rank_bound_checked(i), ".*");
        } else {
            EXPECT_EQ(rank + i, comm.compute_rank_bound_checked(i));
        }
    }
}

TEST_F(CommunicatorTest, RankAdvanceCyclic) {
    Communicator comm;

    for (int i = -(2 * size); i < (2 * size); ++i) {
        EXPECT_EQ((rank + i) % size, comm.compute_rank_circular(i));
    }
}

TEST_F(CommunicatorTest, ValidRank) {
    Communicator comm;

    int mpi_size;
    MPI_Comm_size(comm.mpi_communicator(), &mpi_size);

    for (int i = -(2 * mpi_size); i < (2 * mpi_size); ++i) {
        EXPECT_EQ((i >= 0 && i < mpi_size), comm.is_valid_rank(i));
    }
}
