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

#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <numeric>

struct ExampleTest : ::testing::Test {
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
        recv_buf.resize(static_cast<std::size_t>(size));
    }
    MPI_Gather(&rank, 1, MPI_INT, recv_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::vector<int> expected(static_cast<std::size_t>(size));
        std::iota(expected.begin(), expected.end(), 0);
        EXPECT_THAT(recv_buf, testing::Eq(expected));
    }
}
