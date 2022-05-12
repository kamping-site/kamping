// This file is part of KaMPIng.
//
// Copyright 2022 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPIng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPIng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <gtest/gtest.h>
#include <mpi.h>

#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(BarrierTest, barrier) {
    Communicator comm;

    // One rank calls MPI_Barrier directly, all others call comm.barrier.
    // If comm.barrier() correctly calls MPI_Barrier, this finishes. Otherwise it deadlocks on at least one rank.

    // Use rank 1 as the rank that calls MPI_Barrier directly so when running on 1 rank, we test that comm.barrier()
    // finishes correctly.
    if (comm.rank() == 1) {
        MPI_Barrier(comm.mpi_communicator());
    } else {
        comm.barrier();
    }
}
