// This file is part of KaMPI.ng.
//
// Copyright 2021 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

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
    EXPECT_EQ(comm.root(), 0);
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
    EXPECT_EQ(comm.rank(), 0);

    EXPECT_DEATH(Communicator comm2(MPI_COMM_NULL), ".*");
}

TEST_F(CommunicatorTest, ConstructorWithMPICommunicatorAndRoot) {
    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i < 0 || i >= size) {
            EXPECT_DEATH(Communicator comm(MPI_COMM_WORLD, i), ".*");
            EXPECT_DEATH(Communicator comm(MPI_COMM_NULL, i), ".*");
        } else {
            Communicator comm(MPI_COMM_WORLD, i);
            ASSERT_EQ(comm.root(), i);
            EXPECT_DEATH(Communicator comm2(MPI_COMM_NULL, i), ".*");
        }
    }
}

TEST_F(CommunicatorTest, SetRankBoundCheck) {
  Communicator comm;
    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i < 0 || i >= size) {
          EXPECT_DEATH(comm.root(i), ".*");
        } else {
          comm.root(i);
          EXPECT_EQ(i, comm.root());
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
