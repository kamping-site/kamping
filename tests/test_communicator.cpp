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

TEST_F(CommunicatorTest, RankAdvanceBoundCheck) {
  Communicator comm;

  for (int i = -(2 * size); i < (2 * size); ++i) {
    if (i + rank < 0 || i + rank >= size) {
      EXPECT_DEATH(comm.rank_advance_bound_checked(i), ".*");
    } else {
      EXPECT_EQ(rank + i, comm.rank_advance_bound_checked(i));
    }
  }
}

TEST_F(CommunicatorTest, RankAdvanceCyclic) {
  Communicator comm;

  for (int i = -(2 * size); i < (2 * size); ++i) {
    EXPECT_EQ((rank + i) % size, comm.rank_advance_cyclic(i));
  }
}
