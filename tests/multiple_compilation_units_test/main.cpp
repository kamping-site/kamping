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

/// @file
/// @brief The main file (and first compilation unit) of a test that checks if compiling and running works correctly
/// when linking two compilation units that both use KaMPIng

#include <cstddef>
#include <numeric>

#include <gtest/gtest.h>
#include <mpi.h>

// include all collectives so we can catch errors
#include "./gatherer.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/collectives/barrier.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/collectives/scatter.hpp"
#include "kamping/communicator.hpp"
#include "kamping/environment.hpp"

/// @brief The main function for this tests. Gathers the ranks on the root (done in a different compilation unit), calls
/// a barrier and checks the result
TEST(TwoCompilationUnitsTest, main) {
    using namespace kamping;
    EXPECT_TRUE(mpi_env.initialized());

    Communicator comm;

    Gatherer gatherer;
    auto     gathered_data = gatherer.gather(comm.rank_signed());

    comm.barrier();

    if (comm.rank() == 0) {
        std::vector<int> expected_result(static_cast<size_t>(comm.size()));
        std::iota(expected_result.begin(), expected_result.end(), 0);
        EXPECT_EQ(gathered_data, expected_result);
    } else {
        EXPECT_TRUE(gathered_data.empty());
    }
}
