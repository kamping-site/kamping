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

// overwrite build options and set assertion level to normal, enable exceptions
#undef KAMPING_ASSERTION_LEVEL
#define KAMPING_ASSERTION_LEVEL kamping::assert::normal
#ifndef KAMPING_EXCEPTION_MODE
    #define KAMPING_EXCEPTION_MODE
#endif // KAMPING_EXCEPTION_MODE

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

    EXPECT_THROW(Communicator(MPI_COMM_NULL), assert::KassertException);
}

TEST_F(CommunicatorTest, ConstructorWithMPICommunicatorAndRoot) {
    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i < 0 || i >= size) {
            EXPECT_THROW(Communicator(MPI_COMM_WORLD, i), assert::KassertException);
            EXPECT_THROW(Communicator(MPI_COMM_NULL, i), assert::KassertException);
        } else {
            Communicator comm(MPI_COMM_WORLD, i);
            ASSERT_EQ(comm.root(), i);

            EXPECT_THROW(Communicator(MPI_COMM_NULL, i), assert::KassertException);
        }
    }
}

TEST_F(CommunicatorTest, SetRootkBoundCheck) {
    Communicator comm;
    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i < 0 || i >= size) {
            EXPECT_THROW(comm.root(i), assert::KassertException);
        } else {
            comm.root(i);
            EXPECT_EQ(i, comm.root());
        }
    }
}

TEST_F(CommunicatorTest, RankShiftedChecked) {
    Communicator comm;

    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i + rank < 0 || i + rank >= size) {
            EXPECT_THROW(((void)comm.rank_shifted_checked(i)), assert::KassertException);
        } else {
            EXPECT_EQ(rank + i, comm.rank_shifted_checked(i));
        }
    }
}

TEST_F(CommunicatorTest, RankShiftedeCyclic) {
    Communicator comm;

    for (int i = -(2 * size); i < (2 * size); ++i) {
        EXPECT_EQ((rank + i) % size, comm.rank_shifted_cyclic(i));
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

TEST_F(CommunicatorTest, SplitAndRankConversion) {
    Communicator comm;

    // Test split with any number of reasonable colors
    for (int i = 2; i <= size; ++i) {
        int const color         = rank % i;
        auto      splitted_comm = comm.split(color);
        int const expected_size = (size / i) + ((size % i > rank % i) ? 1 : 0);
        EXPECT_EQ(splitted_comm.size(), expected_size);

        // Check for all rank ids whether they correctly convert to the splitted communicator
        for (int rank_to_test = 0; rank_to_test < size; ++rank_to_test) {
            int const expected_rank_in_splitted_comm = rank_to_test % i == color ? rank_to_test / i : MPI_UNDEFINED;
            EXPECT_EQ(expected_rank_in_splitted_comm, comm.convert_rank_to_communicator(rank_to_test, splitted_comm));
            EXPECT_EQ(expected_rank_in_splitted_comm, splitted_comm.convert_rank_from_communicator(rank_to_test, comm));
            if (expected_rank_in_splitted_comm != MPI_UNDEFINED) {
                EXPECT_EQ(
                    rank_to_test, comm.convert_rank_from_communicator(expected_rank_in_splitted_comm, splitted_comm));
                EXPECT_EQ(
                    rank_to_test, splitted_comm.convert_rank_to_communicator(expected_rank_in_splitted_comm, comm));
            }
        }
    }

    // Test split with any number of reasonable colors and inverse keys
    for (int i = 2; i <= size; ++i) {
        int const color         = rank % i;
        auto      splitted_comm = comm.split(color, size - rank);
        int const expected_size = (size / i) + ((size % i > rank % i) ? 1 : 0);
        EXPECT_EQ(splitted_comm.size(), expected_size);

        int const smaller_ranks_in_split = rank / i;
        int const expected_rank          = expected_size - smaller_ranks_in_split - 1;
        EXPECT_EQ(splitted_comm.rank(), expected_rank);

        // Check for all rank ids whether they correctly convert to the splitted communicator
        for (int rank_to_test = 0; rank_to_test < size; ++rank_to_test) {
            int const expected_rank_rn_splitted_comm =
                rank_to_test % i == color ? expected_size - (rank_to_test / i) - 1 : MPI_UNDEFINED;
            EXPECT_EQ(expected_rank_rn_splitted_comm, comm.convert_rank_to_communicator(rank_to_test, splitted_comm));
            EXPECT_EQ(expected_rank_rn_splitted_comm, splitted_comm.convert_rank_from_communicator(rank_to_test, comm));
            if (expected_rank_rn_splitted_comm != MPI_UNDEFINED) {
                EXPECT_EQ(
                    rank_to_test, comm.convert_rank_from_communicator(expected_rank_rn_splitted_comm, splitted_comm));
                EXPECT_EQ(
                    rank_to_test, splitted_comm.convert_rank_to_communicator(expected_rank_rn_splitted_comm, comm));
            }
        }
    }
}
