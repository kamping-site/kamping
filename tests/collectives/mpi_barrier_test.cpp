
// This file is part of KaMPI.ng.
//
// Copyright 2022 The KaMPI.ng Authors
//
// KaMPI.ng is free software : you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version. KaMPI.ng is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License along with KaMPI.ng.  If not, see
// <https://www.gnu.org/licenses/>.

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "kamping/communicator.hpp"


using namespace ::kamping;
using namespace ::testing;

TEST(BarrierTest, barrier) {
    const uint64_t sleep_for_ms = 10;
    Communicator   comm;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    if (comm.is_root()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_for_ms));
    }
    comm.barrier();
    auto end       = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    EXPECT_GE(time_diff, sleep_for_ms);
}