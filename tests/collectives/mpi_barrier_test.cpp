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

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "kamping/collectives/barrier.hpp"
#include "kamping/communicator.hpp"

using namespace ::kamping;
using namespace ::testing;

TEST(BarrierTest, barrier) {
    Communicator comm;

    // Test the given barrier implementation. Returns true, if the test passes, false otherwise.
    auto test_the_barrier = [&comm](auto barrierImpl, long sleep_for_ms) -> bool {
        // All processes take the current time.
        MPI_Barrier(MPI_COMM_WORLD);
        // If we are unlucky, some processes exit this barrier more than sleep_for_ms after the root rank, which will
        // cause this test to fail, even for a valid barrier implementation.
        auto start = std::chrono::high_resolution_clock::now();
        // Ensure that we start our timer *before* the root goes to sleep
        MPI_Barrier(MPI_COMM_WORLD);

        // The root process sleeps for a predefined amount of time before entering the barrier.
        if (comm.is_root()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_for_ms));
        }
        // All other processes enter the barrier immediately.

        barrierImpl();

        // All processes check if they spent at least the amount of time the root process slept inside the barrier.
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

    // On a single rank, there is no such thing as an _invalid_ barrier implementation (except when something crashes,
    // deadlocks, or does not compile).
    if (comm.size() == 1) {
        // Test that our barrier() compiles, does not crash, and does not deadlock.
        EXPECT_TRUE(test_the_barrier([&comm]() { comm.barrier(); }, 10));
    } else {
        // If the scheduling is such that the non-root processes are not scheduled for longer than the root process
        // sleep()s, a broken barrier implementation might yield a false positive. We therefore have to test multiple
        // sleep durations until the test fails.
        bool test_failed  = false;
        long sleep_for_ms = 10;
        while (!test_failed) {
            test_failed = !test_the_barrier([] { return; }, sleep_for_ms);
            sleep_for_ms *= 2;
            MPI_Barrier(MPI_COMM_WORLD);
        }
        ASSERT_TRUE(test_failed);

        // Even with this empirically determined sleep duration, we still get some false-negative test results for a
        // valid barrier implementation. If the scheduler pauses all non-root processes for longer than sleep_for_ms,
        // between starting the time measurement and entering the (broken) barrier, this test will yield a
        // false-positive. We therefore perform multiple iterations of this test, and then accept or deny the barrier
        // implementation depending on if more tests succeeded or failed.
        const uint32_t num_tries          = 9;
        uint32_t       num_test_succeeded = 0;
        uint32_t       num_test_failed    = 0;
        for (uint32_t i = 0; i < num_tries; ++i) {
            if (test_the_barrier([&comm] { comm.barrier(); }, sleep_for_ms)) {
                ++num_test_succeeded;
            } else {
                ++num_test_failed;
            }
        }
        EXPECT_GT(num_test_succeeded, num_test_failed);

        // This will not correctly detect all broken barrier implementations; e.g. the following would pass:
        // [] { std::this_thread::sleep_for(std::chrono::milliseconds(sleep_for_ms)); }
        // On the other hand, detecting if a given function is a valid barrier implementation is equal to solving the
        // halting problem [1].
        // [1] Rice, H. G. (1953), "Classes of recursively enumerable sets and their decision problems", Transactions of
        // the American Mathematical Society, 74 (2): 358–366, doi:10.1090/s0002-9947-1953-0053041-6, JSTOR 1990888
        // ;-)
    }
}
