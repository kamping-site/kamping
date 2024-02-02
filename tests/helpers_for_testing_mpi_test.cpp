// This file is part of KaMPIng.
//
// Copyright 2024 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

#include "helpers_for_testing.hpp"

TEST(DummyNonBlockOperationTest, basics) {
    testing::DummyNonBlockingOperation op;
    kamping::Request                   req;
    int                                x         = 0;
    int                                completed = false;
    op.start_op(kamping::request(req), kamping::tag(42), kamping::recv_buf(x));
    EXPECT_EQ(x, 0);
    MPI_Test(&req.mpi_request(), &completed, MPI_STATUS_IGNORE);
    EXPECT_EQ(x, 0);
    EXPECT_FALSE(completed);
    op.finish_op();
    MPI_Status status;
    MPI_Test(&req.mpi_request(), &completed, &status);
    EXPECT_TRUE(completed);
    EXPECT_EQ(x, 42);
    EXPECT_EQ(status.MPI_TAG, 42);
}
