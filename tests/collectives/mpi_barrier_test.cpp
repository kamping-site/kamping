
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
    Communicator comm;

    // Test the given barrier implementation. Returns true, if the test passes, false otherwise.
    auto test_the_barrier = [&comm](auto barrierImpl, long sleep_for_ms) -> bool {
        // All processes take the current time.
        MPI_Barrier(MPI_COMM_WORLD);
        //! If we are unlucky, some processes exit this barrier more than sleep_for_ms after the root rank, which will
        //! cause this test to fail, even for a valid barrier implementation.
        auto start = std::chrono::high_resolution_clock::now();

        // The root process sleeps for a predefined amount of time, before entering the barrier.
        if (comm.is_root()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_for_ms));
        }
        // All other processes enter the barrier immediately.

        barrierImpl();

        // All processes check, if they spent at least the amount of time the root process slept inside the barrier.
        auto end                 = std::chrono::high_resolution_clock::now();
        auto time_diff           = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        bool i_slept_long_enough = time_diff >= sleep_for_ms;

        // We want to have the same result on all processes.
        bool everyone_slept_long_enough;
        MPI_Allreduce(
            &i_slept_long_enough,        // send buffer
            &everyone_slept_long_enough, // receive buffer
            1,                           // count
            MPI_CXX_BOOL,                // datatype
            MPI_LAND,                    // operation
            MPI_COMM_WORLD               // communicator
        );
        return everyone_slept_long_enough;
    };

    // It is nonsensical to test a barrier implementation on a single rank.
    if (comm.size() > 1) {
        // If the scheduling is such, that the non-root processes are not scheduled for longer than the root process
        // sleep()s, a broken barrier implementation might yield a false positive. We therefore have to test multiple sleep
        // durations until the test fails.
        bool test_failed  = false;
        long sleep_for_ms = 10;
        while (!test_failed) {
            test_failed = !test_the_barrier([] { return; }, sleep_for_ms);
            sleep_for_ms *= 2;
            MPI_Barrier(MPI_COMM_WORLD);
        }
        ASSERT_TRUE(test_failed);

        // Even with this empirically determined sleep duration, we still get some false-negative test results for a valid
        // barrier implementation. As this test can't be false positive, we can re-run it a given number of times or until
        // it succeeds to get more reliable results. (See also the comment marked with ! above.)
        const uint32_t max_tries      = 8;
        bool           test_succeeded = false;
        for (uint32_t i = 0; i < max_tries && !test_succeeded; ++i) {
            test_succeeded = test_the_barrier([&comm] { comm.barrier(); }, sleep_for_ms);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        EXPECT_TRUE(test_succeeded);

        // This will not correctly detect all broken barrier implementations; e.g. the following would pass:
        // [] { std::this_thread::sleep_for(std::chrono::milliseconds(sleep_for_ms)); }
        // On the other hand, detecting if a given function is a valid barrier implementation is equal to solving the
        // halting problem.
    }
}
