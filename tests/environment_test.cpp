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
#include <mpi.h>

#include "helpers_for_testing.hpp"
#include "kamping/environment.hpp"

using namespace ::kamping;

struct EnvironmentTest : testing::Test {
    void SetUp() override {
        int  flag;
        int* value;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &flag);
        EXPECT_TRUE(flag);
        mpi_tag_ub = *value;
    }

    int mpi_tag_ub;
};

TEST_F(EnvironmentTest, wtime) {
    const std::chrono::milliseconds::rep milliseconds_to_sleep = 10;
    double const                         seconds_to_sleep      = static_cast<double>(milliseconds_to_sleep) / 1000.0;
    // Get the first time from an object
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    double                                            start_time = env.wtime();

    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds_to_sleep));

    // Get the second time from the class to check that wtime is static
    double end_time = Environment<>::wtime();

    EXPECT_GE(end_time, start_time + seconds_to_sleep);
}

TEST_F(EnvironmentTest, wtick) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    double                                            kamping_wtick = env.wtick();
    EXPECT_DOUBLE_EQ(kamping_wtick, MPI_Wtick());

    kamping_wtick = Environment<>::wtick();
    EXPECT_DOUBLE_EQ(kamping_wtick, MPI_Wtick());
}

TEST_F(EnvironmentTest, init) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    EXPECT_TRUE(env.initialized());
    // This should succeed because init checks whether MPI_Init has already been called.
    env.init();
}

TEST_F(EnvironmentTest, init_unchecked) {
    Environment<kamping::InitMPIMode::NoInitFinalize> env;
    EXPECT_TRUE(env.initialized());
    EXPECT_KASSERT_FAILS(env.init_unchecked(), "Trying to call MPI_Init twice");
}

TEST_F(EnvironmentTest, tag_upper_bound) {
    ASSERT_EQ(mpi_env.tag_upper_bound(), mpi_tag_ub);
    ASSERT_GE(mpi_env.tag_upper_bound(), 32767); // the standard requires that MPI_TAG_UB has at least this size
}

TEST_F(EnvironmentTest, is_valid_tag) {
    ASSERT_TRUE(mpi_env.is_valid_tag(0));
    ASSERT_TRUE(mpi_env.is_valid_tag(42));
    ASSERT_TRUE(mpi_env.is_valid_tag(mpi_tag_ub));
    // Avoid signed integer overflow
    if (mpi_tag_ub < std::numeric_limits<decltype(mpi_tag_ub)>::max()) {
        ASSERT_FALSE(mpi_env.is_valid_tag(mpi_tag_ub + 1));
    }

    if (mpi_tag_ub == std::numeric_limits<int>::max()) {
        ASSERT_TRUE(mpi_env.is_valid_tag(std::numeric_limits<int>::max()));
    } else {
        ASSERT_FALSE(mpi_env.is_valid_tag(std::numeric_limits<int>::max()));
    }
    ASSERT_FALSE(mpi_env.is_valid_tag(-1));
    ASSERT_FALSE(mpi_env.is_valid_tag(-42));
    ASSERT_FALSE(mpi_env.is_valid_tag(std::numeric_limits<int>::min()));
}
