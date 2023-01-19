// This file is part of KaMPIng.
//
// Copyright 2021-2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <limits>

#include <gtest/gtest.h>
#include <kassert/kassert.hpp>
#include <mpi.h>

#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

struct CommunicatorTest : Test {
    void SetUp() override {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int  flag;
        int* value;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &flag);
        EXPECT_TRUE(flag);
        mpi_tag_ub = *value;
    }

    int rank;
    int size;
    int mpi_tag_ub;
};

TEST_F(CommunicatorTest, empty_constructor) {
    Communicator comm;

    EXPECT_EQ(comm.mpi_communicator(), MPI_COMM_WORLD);
    EXPECT_EQ(comm.rank(), rank);
    EXPECT_EQ(comm.rank_signed(), rank);
    EXPECT_EQ(comm.size_signed(), size);
    EXPECT_EQ(comm.size(), size);
    EXPECT_EQ(comm.root(), 0);
    EXPECT_EQ(comm.root_signed(), 0);
}

TEST_F(CommunicatorTest, constructor_with_mpi_communicator) {
    Communicator comm(MPI_COMM_SELF);

    int self_rank;
    int self_size;

    MPI_Comm_size(MPI_COMM_SELF, &self_size);
    MPI_Comm_rank(MPI_COMM_SELF, &self_rank);

    EXPECT_EQ(comm.mpi_communicator(), MPI_COMM_SELF);
    EXPECT_EQ(comm.rank_signed(), self_rank);
    EXPECT_EQ(comm.rank(), self_rank);
    EXPECT_EQ(comm.size_signed(), self_size);
    EXPECT_EQ(comm.size(), self_size);
    EXPECT_EQ(comm.rank_signed(), 0);
    EXPECT_EQ(comm.rank(), 0);

    EXPECT_THROW(Communicator(MPI_COMM_NULL), kassert::KassertException);
}

TEST_F(CommunicatorTest, constructor_with_mpi_communicator_and_root) {
    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i < 0 || i >= size) {
            EXPECT_THROW(Communicator(MPI_COMM_WORLD, i), kassert::KassertException);
            EXPECT_THROW(Communicator(MPI_COMM_NULL, i), kassert::KassertException);
        } else {
            Communicator comm(MPI_COMM_WORLD, i);
            ASSERT_EQ(comm.root(), i);

            EXPECT_THROW(Communicator(MPI_COMM_NULL, i), kassert::KassertException);
        }
    }
}

TEST_F(CommunicatorTest, is_root) {
    Communicator comm;
    if (comm.root() == comm.rank()) {
        EXPECT_TRUE(comm.is_root());
    } else {
        EXPECT_FALSE(comm.is_root());
    }

    int const custom_root = comm.size_signed() - 1;
    if (custom_root == comm.rank_signed()) {
        EXPECT_TRUE(comm.is_root(custom_root));
    } else {
        EXPECT_FALSE(comm.is_root(custom_root));
    }
}

TEST_F(CommunicatorTest, set_root_bound_check) {
    Communicator comm;
    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i < 0 || i >= size) {
            EXPECT_THROW(comm.root(i), kassert::KassertException);
        } else {
            comm.root(i);
            EXPECT_EQ(i, comm.root());
            if (i > 0) {
                comm.root(asserting_cast<size_t>(i));
                EXPECT_EQ(i, comm.root());
            }
            if (comm.rank_signed() == i) {
                EXPECT_TRUE(comm.is_root());
            } else {
                EXPECT_FALSE(comm.is_root());
            }
        }
    }
}

TEST_F(CommunicatorTest, set_default_tag) {
    Communicator comm;
    ASSERT_EQ(comm.default_tag(), 0);
    comm.default_tag(1);
    ASSERT_EQ(comm.default_tag(), 1);
    comm.default_tag(23);
    ASSERT_EQ(comm.default_tag(), 23);
    comm.default_tag(mpi_tag_ub);
    ASSERT_EQ(comm.default_tag(), mpi_tag_ub);
    // Avoid signed integer overflow
    if (mpi_tag_ub < std::numeric_limits<decltype(mpi_tag_ub)>::max()) {
        EXPECT_THROW(comm.default_tag(mpi_tag_ub + 1), kassert::KassertException);
    }
    EXPECT_THROW(comm.default_tag(-1), kassert::KassertException);
}

TEST_F(CommunicatorTest, rank_shifted_checked) {
    Communicator comm;

    for (int i = -(2 * size); i < (2 * size); ++i) {
        if (i + rank < 0 || i + rank >= size) {
            EXPECT_THROW(((void)comm.rank_shifted_checked(i)), kassert::KassertException);
        } else {
            EXPECT_EQ(rank + i, comm.rank_shifted_checked(i));
        }
    }
}

TEST_F(CommunicatorTest, rank_shifted_cyclic) {
    Communicator comm;

    for (int i = -(2 * size); i < (2 * size); ++i) {
        EXPECT_EQ((rank + i + 2 * size) % size, comm.rank_shifted_cyclic(i));
    }
}

TEST_F(CommunicatorTest, valid_rank) {
    Communicator comm;

    int mpi_size;
    MPI_Comm_size(comm.mpi_communicator(), &mpi_size);

    for (int i = -(2 * mpi_size); i < (2 * mpi_size); ++i) {
        EXPECT_EQ((i >= 0 && i < mpi_size), comm.is_valid_rank(i));
    }

    for (size_t i = 0; i < (2 * asserting_cast<size_t>(mpi_size)); ++i) {
        EXPECT_EQ(i < asserting_cast<size_t>(mpi_size), comm.is_valid_rank(i));
    }
}

TEST_F(CommunicatorTest, split_and_rank_conversion) {
    Communicator comm;

    // Test split with any number of reasonable colors
    for (int i = 2; i <= size; ++i) {
        int const color         = rank % i;
        auto      splitted_comm = comm.split(color);
        int const expected_size = (size / i) + ((size % i > rank % i) ? 1 : 0);
        EXPECT_EQ(splitted_comm.size(), expected_size);
        EXPECT_EQ(splitted_comm.size_signed(), expected_size);

        // Check for all rank ids whether they correctly convert to the splitted communicator
        for (int rank_to_test = 0; rank_to_test < size; ++rank_to_test) {
            int const expected_rank_in_splitted_comm = rank_to_test % i == color ? rank_to_test / i : MPI_UNDEFINED;
            EXPECT_EQ(expected_rank_in_splitted_comm, comm.convert_rank_to_communicator(rank_to_test, splitted_comm));
            EXPECT_EQ(expected_rank_in_splitted_comm, splitted_comm.convert_rank_from_communicator(rank_to_test, comm));
            if (expected_rank_in_splitted_comm != MPI_UNDEFINED) {
                EXPECT_EQ(
                    rank_to_test,
                    comm.convert_rank_from_communicator(expected_rank_in_splitted_comm, splitted_comm)
                );
                EXPECT_EQ(
                    rank_to_test,
                    splitted_comm.convert_rank_to_communicator(expected_rank_in_splitted_comm, comm)
                );
            }
        }
    }

    // Test split with any number of reasonable colors and inverse keys
    for (int i = 2; i <= size; ++i) {
        int const color         = rank % i;
        auto      splitted_comm = comm.split(color, size - rank);
        int const expected_size = (size / i) + ((size % i > rank % i) ? 1 : 0);
        EXPECT_EQ(splitted_comm.size(), expected_size);
        EXPECT_EQ(splitted_comm.size_signed(), expected_size);

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
                    rank_to_test,
                    comm.convert_rank_from_communicator(expected_rank_rn_splitted_comm, splitted_comm)
                );
                EXPECT_EQ(
                    rank_to_test,
                    splitted_comm.convert_rank_to_communicator(expected_rank_rn_splitted_comm, comm)
                );
            }
        }
    }
}

TEST_F(CommunicatorTest, assignment) {
    // move assignment
    Communicator comm;
    comm = Communicator();

    // copy assignment
    Communicator comm2;
    comm = comm2;
}

TEST_F(CommunicatorTest, comm_world) {
    // These are what comm_world is intended for.
    EXPECT_EQ(comm_world().rank(), rank);
    EXPECT_EQ(comm_world().size(), size);
    EXPECT_EQ(comm_world().rank_signed(), rank);
    EXPECT_EQ(comm_world().size_signed(), size);
}

TEST_F(CommunicatorTest, comm_world_convenience_functions) {
    EXPECT_EQ(world_rank(), rank);
    EXPECT_EQ(world_size(), size);
    EXPECT_EQ(world_rank_signed(), rank);
    EXPECT_EQ(world_size_signed(), size);
}
